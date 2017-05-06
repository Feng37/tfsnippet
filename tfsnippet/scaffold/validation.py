# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import shutil
import warnings
from contextlib import contextmanager
from logging import getLogger

import tensorflow as tf
from tfsnippet.utils import VariableSaver, TemporaryDirectory, makedirs

__all__ = ['early_stopping']


class _EarlyStopping(object):
    """Class to hold the best loss within an early-stopping context.

    Parameters
    ----------
    saver : VariableSaver
        The variable saver for early-stopping.

    best_metric : float
        The initial best loss (default None).

    smaller_is_better : bool
        Whether or not the less, the better loss? (default True)
    """

    def __init__(self, saver, best_metric=None, smaller_is_better=True):
        self._saver = saver
        self._best_metric = best_metric
        self._smaller_is_better = smaller_is_better
        self._ever_updated = False

    def update(self, metric, global_step=None):
        """Update the best metric.

        Parameters
        ----------
        metric : float
            New metric value.

        global_step : int
            Optional global step counter.

        Returns
        -------
        bool
            Whether or not the best loss has been updated?
        """
        self._ever_updated = True
        if self._best_metric is None or \
                (self._smaller_is_better and metric < self._best_metric) or \
                (not self._smaller_is_better and metric > self._best_metric):
            self._saver.save(global_step)
            self._best_metric = metric
            return True
        return False

    @property
    def best_metric(self):
        """Get the current best loss."""
        return self._best_metric

    @property
    def ever_updated(self):
        """Check whether or not `update` method has ever been called."""
        return self._ever_updated


@contextmanager
def early_stopping(param_vars, initial_metric=None, save_dir=None,
                   smaller_is_better=True, restore_on_error=False,
                   cleanup=True, name=None):
    """Open a context to memorize the values of parameters at best metric.

    This method will open a context with an object to memorize the best
    metric for early-stopping.  An example of using this early-stopping
    context is:

        with early_stopping(param_vars) as es:
            ...
            es.update(loss, global_step)
            ...

    Where ``es.update(loss, global_step)`` should cause the parameters to
    be saved on disk if `loss` is better than the current best metric.
    One may also get the best metric via ``es.best_metric``.

    Note that if no loss is given via ``es.update``, then the variables
    would always be restored to their initial states after exiting context.

    Parameters
    ----------
    param_vars : list[tf.Variable] | dict[str, tf.Variable]
        List or dict of variables to be memorized.

        If a dict is specified, the keys of the dict would be used as the
        serializations keys via `VariableSaver`.

    initial_metric : float | tf.Tensor | tf.Variable
        The initial best loss (usually for recovering from previous session).

    save_dir : str
        The directory where to save the variable values.
        If not specified, will use a temporary directory.

    smaller_is_better : bool
        Whether or not the less, the better loss? (default True)

    restore_on_error : bool
        Whether or not to restore the memorized parameters even on error?
        (default False)

    cleanup : bool
        Whether or not to cleanup the saving directory on exit?

        This argument will be ignored if `save_dir` is None, while
        the temporary directory will always be deleted on exit.

    name : str
        Optional name of this scope.

    Yields
    ------
    _EarlyStopping
        The object to receive loss during early-stopping context.
    """
    if not param_vars:
        raise ValueError('`param_vars` must not be empty.')

    if save_dir is None:
        with TemporaryDirectory() as tempdir:
            with early_stopping(param_vars, initial_metric=initial_metric,
                                save_dir=tempdir, cleanup=False,
                                smaller_is_better=smaller_is_better,
                                restore_on_error=restore_on_error,
                                name=name) as es:
                yield es

    else:
        if isinstance(initial_metric, (tf.Tensor, tf.Variable)):
            initial_metric = initial_metric.eval()

        with tf.name_scope(name):
            saver = VariableSaver(param_vars, save_dir)
            save_dir = os.path.abspath(save_dir)
            makedirs(save_dir, exist_ok=True)

            es = _EarlyStopping(saver,
                                best_metric=initial_metric,
                                smaller_is_better=smaller_is_better)
            saver.save()

            try:
                yield es
            except Exception as ex:
                if isinstance(ex, KeyboardInterrupt) or restore_on_error:
                    saver.restore()
                raise
            else:
                saver.restore()
            finally:
                if cleanup:
                    try:
                        if os.path.exists(save_dir):
                            shutil.rmtree(save_dir)
                    except Exception:
                        getLogger(__name__).error(
                            'Failed to cleanup validation save dir %r.',
                            save_dir, exc_info=True
                        )
                if not es.ever_updated:
                    warnings.warn(
                        'Early-stopping metric has never been updated. '
                        'Did you forget to add corresponding metric?'
                    )
