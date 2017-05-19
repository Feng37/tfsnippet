# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tfsnippet.utils import (get_preferred_tensor_dtype,
                             reopen_variable_scope,
                             is_deterministic_shape,
                             ReshapeHelper)
from .base import Distribution

__all__ = [
    'Categorical'
]


class _BaseCategorical(Distribution):
    """Base categorical distribution class.

    Parameters
    ----------
    logits : tf.Tensor | np.ndarray
        A float 1-d tensor of shape (..., n_categories), which is the
        un-normalized log-odds of probabilities of the categories.

    probs : tf.Tensor | np.ndarray
        A float 1-d tensor of shape (..., n_categories), which is the
        normalized probabilities of the categories.

        One and only one of `logits` and `probs` should be specified.
        The relationship between these two arguments, if given each other,
        is stated as follows:

            .. math::
                \\begin{aligned}
                    \\text{logits} &= \\log (\\text{probs}) \\\\
                     \\text{probs} &= \\text{softmax} (\\text{logits})
                \\end{aligned}

    group_event_ndims : int
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

    name : str
        Name of this normal distribution.

    default_name : str
        Default name of this normal distribution.
    """

    def __init__(self, logits=None, probs=None, group_event_ndims=None,
                 name=None, default_name=None):
        # check the arguments
        if (logits is None and probs is None) or \
                (logits is not None and probs is not None):
            raise ValueError('One and only one of `logits`, `probs` should '
                             'be specified.')

        if logits is not None:
            param_dtype = get_preferred_tensor_dtype(logits)
        else:
            param_dtype = get_preferred_tensor_dtype(probs)
        if not param_dtype.is_floating:
            raise TypeError('Categorical distribution parameters must be float '
                            'numbers.')

        super(_BaseCategorical, self).__init__(
            group_event_ndims=group_event_ndims,
            name=name,
            default_name=default_name
        )

        with reopen_variable_scope(self.variable_scope):
            with tf.name_scope('init'):
                # obtain parameter tensors
                if logits is not None:
                    logits = tf.convert_to_tensor(logits, dtype=param_dtype)
                    probs = tf.nn.softmax(logits, name='probs_given_logits')
                    self._probs_is_derived = True
                else:
                    probs = tf.convert_to_tensor(probs, dtype=param_dtype)
                    logits = tf.log(probs, name='logits_given_probs')
                    self._probs_is_derived = False
                self._logits = logits
                self._probs = probs

                # derive the shape and data types of parameters
                logits_shape = logits.get_shape()
                self._static_batch_shape = logits_shape[:-1]
                if is_deterministic_shape(logits_shape[:-1]):
                    self._dynamic_batch_shape = tf.constant(
                        logits_shape[:-1].as_list(),
                        dtype=tf.int32
                    )
                else:
                    self._dynamic_batch_shape = tf.shape(logits)[:-1]

    @property
    def param_dtype(self):
        return self._logits.dtype

    @property
    def is_continuous(self):
        return False

    @property
    def is_reparameterized(self):
        return False

    @property
    def dynamic_batch_shape(self):
        return self._dynamic_batch_shape

    @property
    def static_batch_shape(self):
        return self._static_batch_shape

    @property
    def logits(self):
        """Get the un-normalized log-odds of Categorical distribution."""
        return self._logits

    @property
    def probs(self):
        """Get the probabilities of Categorical distribution."""
        return self._probs

    def _sample_sparse(self, sample_shape=(), dtype=None):
        # check the arguments of `sample_shape`
        helper = ReshapeHelper(allow_negative_one=False).add(sample_shape)
        static_sample_shape = helper.get_static_shape()
        dynamic_sample_shape = helper.get_dynamic_shape()
        n_samples = (
            np.prod(static_sample_shape.as_list())
            if static_sample_shape.is_fully_defined()
            else tf.reduce_prod(dynamic_sample_shape)
        )

        # derive the samples
        if self.logits.get_shape().ndims == 2:
            logits_2d = self.logits
        else:
            logits_2d = (
                ReshapeHelper().
                    add(-1).
                    add_template(self.logits, lambda s: s[-1]).
                    reshape(self.logits)
            )
        samples = tf.multinomial(logits_2d, n_samples)
        if dtype is not None:
            samples = tf.cast(samples, dtype=dtype)
        samples = (
            ReshapeHelper().
                add(sample_shape).
                add_template(self.logits, lambda s: s[:-1]).
                reshape(tf.transpose(samples))
        )

        return samples

    def _analytic_kld(self, other):
        if isinstance(other, _BaseCategorical):
            # sum(p * log(p / q))
            probs1 = self.probs
            if self._probs_is_derived:
                log_probs1 = tf.nn.log_softmax(self.logits)
            else:
                log_probs1 = tf.log(self.probs)

            if other._probs_is_derived:
                log_probs2 = tf.nn.log_softmax(other.logits)
            else:
                log_probs2 = tf.log(other.probs)

            return tf.reduce_sum(
                probs1 * (log_probs1 - log_probs2), axis=-1)


class Categorical(_BaseCategorical):
    """Categorical distribution class.

    Parameters
    ----------
    logits : tf.Tensor | np.ndarray
        A float 1-d tensor of shape (..., n_categories), which is the
        un-normalized log-odds of probabilities of the categories.

    probs : tf.Tensor | np.ndarray
        A float 1-d tensor of shape (..., n_categories), which is the
        normalized probabilities of the categories.

        One and only one of `logits` and `probs` should be specified.
        The relationship between these two arguments, if given each other,
        is stated as follows:

            .. math::
                \\begin{aligned}
                    \\text{logits} &= \\log (\\text{probs}) \\\\
                     \\text{probs} &= \\text{softmax} (\\text{logits})
                \\end{aligned}

    dtype : tf.DType | np.dtype | str
        The data type of samples from the distribution. (default is `tf.int32`)

    group_event_ndims : int
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

    name : str
        Name of this normal distribution.

    default_name : str
        Default name of this normal distribution.
    """

    def __init__(self, logits=None, probs=None, dtype=None,
                 group_event_ndims=None, name=None, default_name=None):
        if dtype is None:
            dtype = tf.int32
        else:
            dtype = tf.as_dtype(dtype)

        super(Categorical, self).__init__(
            logits=logits,
            probs=probs,
            group_event_ndims=group_event_ndims,
            name=name,
            default_name=default_name
        )
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def dynamic_value_shape(self):
        return tf.constant([], dtype=tf.int32)

    @property
    def static_value_shape(self):
        return tf.TensorShape([])

    def _sample(self, sample_shape=()):
        return self._sample_sparse(sample_shape, self.dtype)

    def _log_prob(self, x):
        x = tf.convert_to_tensor(x)
        if not x.dtype.is_integer:
            x = tf.cast(x, dtype=tf.int32)

        if self.logits.get_shape()[:-1] == x.get_shape():
            logits = self.logits
        else:
            logits = self.logits * tf.ones_like(
                tf.expand_dims(x, -1), dtype=self.logits.dtype)
            logits_shape = tf.shape(logits)[:-1]
            x *= tf.ones(logits_shape, dtype=x.dtype)
            x.set_shape(tf.TensorShape(logits.get_shape()[:-1]))

        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=x, logits=logits,
        )
