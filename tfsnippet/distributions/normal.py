# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tfsnippet.utils import (open_variable_scope, get_preferred_tensor_dtype,
                             ReshapeHelper)
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
        Should be broadcastable to match `mean`.

        Note that the range of `stddev` is :math:`(0, \\infty)`.

    logstd : tf.Tensor | np.ndarray | float
        The log standard derivation of the Normal distribution.
        Should be broadcastable to match `mean`.

        If `stddev` is specified, then `logstd` will be ignored.
        Note that the range of `logstd` is :math:`(-\\infty, \\infty)`.

    group_event_ndims : int
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

    name : str
        Name of this normal distribution.

    default_name : str
        Default name of this normal distribution.
    """

    def __init__(self, mean, stddev=None, logstd=None, group_event_ndims=None,
                 name=None, default_name=None):
        # check the arguments
        if stddev is None and logstd is None:
            raise ValueError('At least one of `stddev`, `logstd` should be '
                             'specified.')
        dtype = get_preferred_tensor_dtype(mean)
        if not dtype.is_floating:
            raise TypeError('Normal distribution parameters must be float '
                            'numbers.')

        super(Normal, self).__init__(group_event_ndims=group_event_ndims,
                                     name=name,
                                     default_name=default_name)

        with open_variable_scope(self.variable_scope, unique_name_scope=False):
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
                    self._stddev = tf.exp(self._stdx, name='stddev')
                    self._logstd = tf.identity(self._stdx, name='logstd')
                    self._var = tf.exp(
                        tf.constant(2., dtype=dtype) * self._logstd,
                        name='variance'
                    )
                    self._precision = tf.exp(
                        tf.constant(-2., dtype=dtype) * self._logstd,
                        name='precision'
                    )
                else:
                    self._stddev = tf.identity(self._stdx, name='stddev')
                    self._logstd = tf.log(self._stdx, name='logstd')
                    self._var = tf.square(self._stddev, name='variance')
                    self._precision = tf.divide(
                        tf.constant(1., dtype=dtype), self._var,
                        name='precision'
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
    def dynamic_batch_shape(self):
        return self._dynamic_batch_shape

    @property
    def static_batch_shape(self):
        return self._static_batch_shape

    @property
    def dynamic_value_shape(self):
        return ()

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

    def sample(self, sample_shape=(), name=None):
        with tf.name_scope(name, default_name='sample'):
            # check the arguments of `sample_shape`
            helper = ReshapeHelper(allow_negative_one=False).add(sample_shape)
            static_sample_shape = helper.get_static_shape()
            dynamic_sample_shape = helper.get_dynamic_shape()

            # derive the sampler
            static_shape = (
                tf.TensorShape(static_sample_shape).
                    concatenate(self.static_batch_shape)
            )
            dynamic_shape = tf.concat(
                [dynamic_sample_shape, self.dynamic_batch_shape],
                axis=0
            )
            samples = self.mean + self.stddev * (
                tf.random_normal(dynamic_shape, dtype=self.dtype)
            )
            samples.set_shape(static_shape)
            return samples

    def _log_prob(self, x):
        c = tf.constant(-0.5 * np.log(2 * np.pi), dtype=self.dtype)
        return (
            c - self.logstd -
            tf.constant(0.5, dtype=self.dtype) * (
                self.precision * tf.square(x - self.mean))
        )

    def analytic_kld(self, other, name=None):
        if isinstance(other, Normal):
            with tf.name_scope(name, default_name='analytic_kld'):
                return tf.constant(0.5, dtype=self.dtype) * (
                    self.var * other.precision +
                    tf.square(other.mean - self.mean) * other.precision +
                    other.logvar - self.logvar - 1
                )
        raise NotImplementedError()
