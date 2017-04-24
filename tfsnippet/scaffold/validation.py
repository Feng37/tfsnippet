# -*- coding: utf-8 -*-
import copy
import os
import shutil
from contextlib import contextmanager
from logging import getLogger

import six
import numpy as np
import tensorflow as tf

from tfsnippet.utils import (get_default_session_or_error,
                             minibatch_slices_iterator,
                             TemporaryDirectory,
                             VariableSaver,
                             try_get_variable_value,
                             ScopedObject,
                             open_variable_scope,
                             makedirs)

__all__ = ['LossValidator']


class LossValidator(ScopedObject):
    """Class to help do validation on loss in training process.
    
    Parameters
    ----------
    inputs : collections.Iterable[tf.Tensor]
        The input tensors for evaluating the validation loss.
        
    valid_loss : tf.Tensor
        The validation loss tensor.  It must be a scalar tensor.

    global_step : tf.Tensor | tf.Variable
        Tensor to track the global step.
        
    managed_vars : dict[str, tf.Variable]
        Dict of variables regularized by best validation loss.
        The keys of the dict would be used as serialization keys.
        
    name : str
        Name of this loss validator.
        
    default_name : str
        Default name of this loss validator.
    """

    BEST_LOSS_SAVE_KEY = 'best_validation_loss'

    def __init__(self, inputs, valid_loss, global_step=None, managed_vars=None,
                 name=None, default_name=None):
        if not isinstance(valid_loss, tf.Tensor) or \
                len(valid_loss.get_shape()) != 0:
            raise TypeError('`valid_loss` is expected to be a scalar tensor, '
                            'but got %r.' % (valid_loss,))
        if managed_vars is not None:
            if self.BEST_LOSS_SAVE_KEY in managed_vars:
                raise KeyError(
                    'Variable name %r is reserved.' %
                    self.BEST_LOSS_SAVE_KEY
                )
        super(LossValidator, self).__init__(name, default_name)

        # memorize the arguments
        inputs = tuple(inputs)
        if valid_loss in inputs:
            valid_loss = tf.identity(valid_loss)

        self._inputs = inputs
        self._valid_loss = valid_loss
        self._global_step = global_step
        self._managed_vars = managed_vars

        # create the variable, the tensors and the operations to track
        # best validation loss
        with open_variable_scope(self.variable_scope):
            self._best_loss = tf.get_variable(
                'best_loss',
                initializer=np.nan,
                dtype=valid_loss.dtype.base_dtype,
                trainable=False
            )
            self._best_loss_ph = tf.placeholder(
                dtype=self._best_loss.dtype.base_dtype,
                shape=(),
                name='best_loss_ph'
            )
            self._best_loss_assign_op = tf.assign(
                self._best_loss, self._best_loss_ph)

        if self._managed_vars is not None:
            self._managed_vars = copy.copy(self._managed_vars)
            self._managed_vars[self.BEST_LOSS_SAVE_KEY] = self._best_loss

        # saver to track the variables for best validation loss,
        # as well as the best validation loss value
        self._context_entered = False
        self._saver = None              # type: VariableSaver
        self._best_loss_val = None      # type: float

    def _merge_feed_dict(self, data, feed_dict):
        ret = {t: v for t, v in zip(self._inputs, data)}
        if feed_dict:
            for k, v in six.iteritems(feed_dict):
                if k not in ret:
                    ret[k] = v
        return ret

    def _compute_loss(self, data, batch_size, feed_dict):
        data = tuple(data)
        if len(data) != len(self._inputs):
            raise ValueError(
                'The validator requires exactly %d input tensors but %d '
                'input arrays are given.' % (len(self._inputs), len(data))
            )
        data_length = None
        if data:
            data_length = len(data[0])
            for d in data[1:]:
                if len(d) != data_length:
                    raise ValueError('Arrays are not in the same length.')

        # get the session to run validation
        session = get_default_session_or_error()

        # if data is not required for making the validation, then just
        # run and get the validation loss
        if not data:
            loss = session.run(self._valid_loss, feed_dict=feed_dict)

        # if the mini-batches is not enabled, compute in one pass.
        elif batch_size is None:
            if not data_length:
                loss = 0.0
            else:
                feed_dict = self._merge_feed_dict(data, feed_dict)
                loss = session.run(self._valid_loss, feed_dict=feed_dict)

        # otherwise prepare for computing the loss in mini-batches.
        else:
            batch_count = (data_length + batch_size - 1) // batch_size
            if batch_count > 0:
                losses = np.zeros(shape=[batch_count], dtype=np.float64)
                batch_slices = minibatch_slices_iterator(
                    data_length, batch_size, ignore_incomplete_batch=False)
                for i, s in enumerate(batch_slices):
                    batch = tuple(d[s] for d in data)
                    feed_dict = self._merge_feed_dict(batch, feed_dict)
                    losses[i] = session.run(self._valid_loss,
                                            feed_dict=feed_dict)

                # merge the loss of each mini-batches
                last_batch_size = data_length % batch_size
                if last_batch_size == 0:
                    loss = np.average(losses)
                else:
                    if batch_count > 1:
                        loss = np.average(losses[:-1])
                        front_portion = batch_count - 1.
                        last_portion = float(last_batch_size) / batch_size
                        total_portion = front_portion + last_portion
                        loss = (loss * (front_portion / total_portion) +
                                losses[-1] * (last_portion / total_portion))
                    else:
                        loss = losses[-1]
            else:
                loss = 0.

        return loss

    def _update_best_loss(self, loss):
        sess = get_default_session_or_error()
        sess.run(self._best_loss_assign_op, feed_dict={
            self._best_loss_ph: loss
        })
        if self._saver is not None:
            self._saver.save(self._global_step)
        self._best_loss_val = loss

    def run(self, data, batch_size=None, feed_dict=None):
        """Do validation with specified `data`.
        
        If there's any `keep_best` context open, then this method will
        save the managed variables if best validation loss is updated.

        Parameters
        ----------
        data : collections.Iterable[np.ndarray]
            The validation data arrays.

        batch_size : int
            If specified, will compute the validation loss in mini-batches.
        
        feed_dict : dict[tf.Tensor, any]
            Additional feed dict other than those specified by `data`.

        Returns
        -------
        float
            The averaged validation loss for given data. 
        """
        loss = self._compute_loss(data, batch_size, feed_dict)
        if self._best_loss_val is None or loss < self._best_loss_val:
            self._update_best_loss(loss)
        return loss

    @contextmanager
    def keep_best(self, save_dir=None, cleanup=True):
        """Open a scoped context to memorize variables at best validation loss.
        
        Parameters
        ----------
        save_dir : str
            The directory where to save the variable values.
            If not specified, will use a temporary directory.
            
        cleanup : bool
            Whether or not to cleanup the saving directory on exit?
            
            This argument will be ignored if `save_dir` is None, while
            the temporary directory will always be deleted on exit.
        """
        if self._context_entered:
            raise RuntimeError('`keep_best` context has been entered.')

        if not self._managed_vars:
            # no variables should be memorized, just enter an empty context
            yield

        elif save_dir is None:
            # open a temporary directory as `save_dir`.
            with TemporaryDirectory() as tempdir, \
                    self.keep_best(tempdir, cleanup=False):
                yield

        else:
            # backup the old `best_loss` and `saver` in case we are in
            # nested `keep_best` context
            save_dir = os.path.abspath(save_dir)
            makedirs(save_dir, exist_ok=True)

            try:
                # restore the best loss from variable, in case it is
                # recovered from checkpoint
                self._best_loss_val = try_get_variable_value(self._best_loss)
                if self._best_loss_val is not None and \
                        np.isnan(self._best_loss_val):
                    self._best_loss_val = None
                with open_variable_scope(self.variable_scope):
                    self._saver = VariableSaver(
                        self._managed_vars, save_dir, save_meta=False)

                # go into to the context
                self._context_entered = True
                yield
            finally:
                self._context_entered = False
                try:
                    # restore the variables at best validation loss
                    if self._saver is not None:
                        self._saver.restore()
                finally:
                    self._saver = None
                    self._best_loss_val = None
                    if cleanup:
                        try:
                            if os.path.exists(save_dir):
                                shutil.rmtree(save_dir)
                        except Exception:
                            getLogger(__name__).error(
                                'Failed to cleanup validation save dir %r.',
                                save_dir, exc_info=True
                            )
