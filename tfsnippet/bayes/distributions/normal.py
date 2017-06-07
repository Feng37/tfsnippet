# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tfsnippet.utils import (reopen_variable_scope, get_preferred_tensor_dtype,
                             is_deterministic_shape)
from .base import Distribution

__all__ = ['Normal']


class Normal(Distribution):
    """Univariate normal distribution.

    Parameters
    ----------
    mean : tf.Tensor | np.ndarray | float
        The mean of the Normal distribution.
        Should be broadcastable to match `stddev`.

    stddev : tf.Tensor | np.ndarray | float
        The standard derivation of the Normal distribution.
        Should be broadcastable to match `mean`.  Also, the data
        type of `stddev` will be casted to that of `mean`.

        Note that the range of `stddev` is :math:`(0, \\infty)`.

    logstd : tf.Tensor | np.ndarray | float
        The log standard derivation of the Normal distribution.
        Should be broadcastable to match `mean`.  Also, the data
        type of `stddev` will be casted to that of `mean`.

        One and only one of `stddev` and `logstd` should be specified.
        Note that the range of `logstd` is :math:`(-\\infty, \\infty)`.

    group_event_ndims : int | tf.Tensor
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

    check_numerics : bool
        Whether or not to check numerical issues? (default False)

    name : str
        Name of this normal distribution.

    default_name : str
        Default name of this normal distribution.
    """

    def __init__(self, mean, stddev=None, logstd=None, group_event_ndims=None,
                 check_numerics=False, name=None, default_name=None):
        # check the arguments
        if (stddev is None and logstd is None) or \
                (stddev is not None and logstd is not None):
            raise ValueError('One and only one of `stddev`, `logstd` should '
                             'be specified.')
        dtype = get_preferred_tensor_dtype(mean)
        if not dtype.is_floating:
            raise TypeError('Normal distribution parameters must be float '
                            'numbers.')

        super(Normal, self).__init__(group_event_ndims=group_event_ndims,
                                     check_numerics=check_numerics,
                                     name=name,
                                     default_name=default_name)

        with reopen_variable_scope(self.variable_scope):
            with tf.name_scope('init'):
                # obtain parameter tensors
                mean = tf.convert_to_tensor(mean, dtype=dtype)
                if stddev is not None:
                    stddev = tf.convert_to_tensor(stddev, dtype=dtype)
                    self._stdx = stddev
                    self._stdx_is_log = False
                else:
                    logstd = tf.convert_to_tensor(logstd, dtype=dtype)
                    self._stdx = logstd
                    self._stdx_is_log = True

                # check the shape and data types of parameters
                self._mean = mean
                try:
                    self._static_batch_shape = tf.broadcast_static_shape(
                        self._mean.get_shape(),
                        self._stdx.get_shape()
                    )
                except ValueError:
                    raise ValueError(
                        '`mean` and `stddev`/`logstd` should be '
                        'broadcastable to match each other (%r vs %r).' %
                        (self._mean.get_shape(), self._stdx.get_shape())
                    )
                self._dynamic_batch_shape = tf.broadcast_dynamic_shape(
                    tf.shape(self._mean), tf.shape(self._stdx)
                )

                # derive the attributes of this Normal distribution
                if self._stdx_is_log:
                    self._stddev = self._check_numerics(
                        tf.exp(self._stdx, name='stddev'), 'stddev')
                    self._logstd = self._stdx
                    self._var = self._check_numerics(
                        tf.exp(
                            tf.constant(2., dtype=dtype) * self._logstd,
                            name='variance'
                        ),
                        'variance'
                    )
                    self._precision = self._check_numerics(
                        tf.exp(
                            tf.constant(-2., dtype=dtype) * self._logstd,
                            name='precision'
                        ),
                        'precision'
                    )
                else:
                    self._stddev = self._stdx
                    self._logstd = self._check_numerics(
                        tf.log(self._stdx, name='logstd'), 'logstd')
                    self._var = tf.square(self._stddev, name='variance')
                    self._precision = self._check_numerics(
                        tf.divide(
                            tf.constant(1., dtype=dtype), self._var,
                            name='precision'
                        ),
                        'precision'
                    )
                self._logvar = tf.multiply(
                    tf.constant(2., dtype=dtype), self._logstd,
                    name='logvar'
                )
                self._log_prec = tf.negative(self._logvar, name='log_precision')

    @property
    def dtype(self):
        return self._mean.dtype.base_dtype

    @property
    def param_dtype(self):
        return self._mean.dtype.base_dtype

    @property
    def is_continuous(self):
        return True

    @property
    def is_reparameterized(self):
        return True

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
    def mean(self):
        """Get the mean of Normal distribution."""
        return self._mean

    @property
    def stddev(self):
        """Get the standard derivation of Normal distribution."""
        return self._stddev

    @property
    def logstd(self):
        """Get the log standard derivation of Normal distribution."""
        return self._logstd

    @property
    def var(self):
        """Get the variance of Normal distribution."""
        return self._var

    @property
    def logvar(self):
        """Get the log-variance of Normal distribution."""
        return self._logvar

    @property
    def precision(self):
        """Get the precision of Normal distribution."""
        return self._precision

    @property
    def log_precision(self):
        """Get the log-precision of Normal distribution."""
        return self._log_prec

    def _sample_n(self, n):
        # compute the shape of samples
        if is_deterministic_shape([n]):
            static_shape = tf.TensorShape([n])
        else:
            static_shape = tf.TensorShape([None])
        static_shape = static_shape.concatenate(self.static_batch_shape)
        dynamic_shape = tf.concat([[n], self.dynamic_batch_shape], axis=0)

        # derive the samples
        samples = self.mean + self.stddev * (
            tf.random_normal(dynamic_shape, dtype=self.dtype)
        )
        samples.set_shape(static_shape)
        return samples

    def _log_prob(self, x):
        x = tf.convert_to_tensor(x, dtype=self.param_dtype)
        c = tf.constant(-0.5 * np.log(2 * np.pi), dtype=self.dtype)
        return (
            c - self.logstd -
            tf.constant(0.5, dtype=self.dtype) * (
                self.precision * tf.square(x - self.mean))
        )

    def _analytic_kld(self, other):
        if isinstance(other, Normal):
            return tf.constant(0.5, dtype=self.dtype) * (
                self.var * other.precision +
                tf.square(other.mean - self.mean) * other.precision +
                other.logvar - self.logvar - 1
            )
