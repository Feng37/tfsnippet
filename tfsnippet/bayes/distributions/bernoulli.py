# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.utils import (get_preferred_tensor_dtype,
                             reopen_variable_scope,
                             is_deterministic_shape,
                             maybe_explicit_broadcast)
from .base import Distribution

__all__ = ['Bernoulli']


class Bernoulli(Distribution):
    """Bernoulli distribution.

    Parameters
    ----------
    logits : tf.Tensor | np.ndarray | float
        A float tensor, which is the log-odds of probabilities of being 1.
        Note that the range of `logits` is :math:`(-\\infty, \\infty)`.

    probs : tf.Tensor | np.ndarray | float
        A float tensor, which is the probabilities of being 1.

        One and only one of `logits` and `p` should be specified.
        The relationship between Bernoulli `p` and `logits` are:

            .. math::
                \\begin{aligned}
                    \\text{logits} &= \\log \\frac{p}{1-p} \\\\
                                 p &= \\frac{1}{1 + \\exp(-\\text{logits})}
                \\end{aligned}

        Note that the range of `probs` is :math:`[0, 1]`.

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
            raise TypeError('Bernoulli distribution parameters must be float '
                            'numbers.')
        if dtype is None:
            dtype = tf.int32
        else:
            dtype = tf.as_dtype(dtype)

        super(Bernoulli, self).__init__(
            group_event_ndims=group_event_ndims,
            name=name,
            default_name=default_name,
        )

        with reopen_variable_scope(self.variable_scope):
            with tf.name_scope('init'):
                # obtain parameter tensors
                if logits is not None:
                    logits = tf.convert_to_tensor(logits, dtype=param_dtype,
                                                  name='logits')
                    probs = tf.nn.sigmoid(logits, 'probs')
                    probs_is_derived = True
                else:
                    probs = tf.convert_to_tensor(probs, dtype=param_dtype,
                                                 name='probs')
                    logits = tf.subtract(
                        tf.log(probs), tf.log1p(-probs),
                        name='logits'
                    )
                    probs_is_derived = False

                # derive the shape and data types of parameters
                logits_shape = logits.get_shape()
                self._logits = logits
                self._static_batch_shape = logits_shape
                if is_deterministic_shape(logits_shape):
                    self._dynamic_batch_shape = tf.constant(
                        logits_shape.as_list(),
                        dtype=tf.int32
                    )
                else:
                    self._dynamic_batch_shape = tf.shape(logits)

                # derive various distribution attributes
                self._probs = probs
                self._probs_is_derived = probs_is_derived

                # set other attributes
                self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def param_dtype(self):
        return self._logits.dtype

    @property
    def is_continuous(self):
        return False

    @property
    def is_reparameterized(self):
        return True

    @property
    def is_enumerable(self):
        return True

    @property
    def enum_value_count(self):
        return 2

    @property
    def dynamic_batch_shape(self):
        return self._dynamic_batch_shape

    @property
    def static_batch_shape(self):
        return self._static_batch_shape

    @property
    def dynamic_value_shape(self):
        return tf.constant([], dtype=tf.int32)

    @property
    def static_value_shape(self):
        return tf.TensorShape([])

    @property
    def logits(self):
        """Get the logits of Bernoulli distribution."""
        return self._logits

    @property
    def mean(self):
        """Get the mean of Bernoulli distribution."""
        return self._probs

    @property
    def probs(self):
        """Get the probabilities of being 1."""
        return self._probs

    def _sample_n(self, n):
        # compute the shape of samples
        if is_deterministic_shape([n]):
            static_shape = tf.TensorShape([n])
        else:
            static_shape = tf.TensorShape([None])
        static_shape = static_shape.concatenate(self.static_batch_shape)
        dynamic_shape = tf.concat([[n], self.dynamic_batch_shape], axis=0)

        # derive the samples
        uniform_samples = tf.random_uniform(
            dynamic_shape, minval=0., maxval=1., dtype=self.param_dtype)
        samples = tf.cast(
            tf.less(uniform_samples, self.probs),
            dtype=self.dtype
        )
        samples.set_shape(static_shape)
        return samples

    def _enum_observe(self):
        # reshape the enumerated values to match the batch shape.
        reshape_shape = tf.concat(
            [[2], tf.ones(tf.stack([tf.size(self.dynamic_batch_shape)]),
                          dtype=tf.int32)],
            axis=0
        )
        samples = tf.reshape(tf.constant([0, 1], dtype=self.dtype),
                             reshape_shape)

        # tile the enumerated values along batch shape
        tile_shape = tf.concat(
            [[1], self.dynamic_batch_shape], axis=0)
        samples = tf.tile(samples, tile_shape)

        # fix the static shape of the samples
        static_shape = tf.TensorShape([2])
        samples.set_shape(static_shape.concatenate(self.static_batch_shape))
        return samples

    def _log_prob(self, x):
        x = tf.cast(x, dtype=self.param_dtype)
        if self._probs_is_derived:
            logits = self.logits
            x, logits = maybe_explicit_broadcast(x, logits)
            return -tf.nn.sigmoid_cross_entropy_with_logits(
                labels=x, logits=logits
            )
        else:
            # TODO: check whether this is better than using derived logits.
            log_p = tf.log(self.probs)
            log_one_minus_p = tf.log1p(-self.probs)
            return x * log_p + (1. - x) * log_one_minus_p

    def _analytic_kld(self, other):
        def get_log_probs(o):
            if o._probs_is_derived:
                log_p = -tf.nn.softplus(-o.logits)
                log_one_minus_p = -tf.nn.softplus(o.logits)
            else:
                log_p = tf.log(o.probs)
                log_one_minus_p = tf.log1p(-o.probs)
            return log_p, log_one_minus_p

        if isinstance(other, Bernoulli):
            p = self.probs
            one_minus_p = (
                tf.nn.sigmoid(-self.logits)
                if self._probs_is_derived
                else 1. - self.probs
            )
            log_p, log_one_minus_p = get_log_probs(self)
            log_q, log_one_minus_q = get_log_probs(other)
            return (
                p * (log_p - log_q) +
                one_minus_p * (log_one_minus_p - log_one_minus_q)
            )
