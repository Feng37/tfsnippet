# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tfsnippet.utils import floatx, is_integer
from .base import Distribution

__all__ = ['Normal']


class Normal(Distribution):
    """Univariate normal distribution.

    Parameters
    ----------
    mean : tf.Tensor
        The mean of the Normal distribution.
        Should be broadcastable to match `stddev`.

    stddev : tf.Tensor
        The standard derivation of the Normal distribution.
        Should be broadcastable to match `mean`.

    logstd : tf.Tensor
        The log standard derivation of the Normal distribution.
        Should be broadcastable to match `mean`.

        If `stddev` is specified, then `logstd` will be ignored.
        
    name : str
        Optional name of this normal distribution.
    """

    def __init__(self, mean, stddev=None, logstd=None, name=None):
        stdx = stddev if stddev is not None else logstd
        if stdx is None:
            raise ValueError('One of `stddev`, `logstd` should be specified.')
        super(Normal, self).__init__(name=name)

        with tf.name_scope(self.name_scope), tf.name_scope('init'):
            # check the shape of parameters
            self._mean = tf.convert_to_tensor(mean, dtype=floatx())
            self._stdx = tf.convert_to_tensor(stdx, dtype=floatx())
            if stddev is None:
                self._stdx_is_log = True
                self._stddev = tf.exp(self._stdx, name='stddev')
                self._logstd = tf.identity(self._stdx, name='logstd')
                self._var = tf.exp(2. * self._logstd, name='variance')
                self._precision = tf.exp(-2. * self._logstd, name='precision')
            else:
                self._stdx_is_log = False
                self._stddev = tf.identity(self._stdx, name='stddev')
                self._logstd = tf.log(self._stdx, name='logstd')
                self._var = tf.square(self._stddev, name='variance')
                self._precision = tf.divide(1., self._var, name='precision')
            self._logvar = tf.multiply(2., self._logstd, name='logvar')
            self._log_prec = tf.negative(self._logvar, name='log_precision')
            try:
                tf.broadcast_static_shape(self._mean.get_shape(),
                                          self._stdx.get_shape())
            except ValueError:
                raise ValueError(
                    '`mean` and `stddev`/`logstd` should be '
                    'broadcastable to match each other (%r vs %r).' %
                    (self._mean.get_shape(), self._stdx.get_shape())
                )

            # get the shape of samples
            self._static_batch_shape = tf.broadcast_static_shape(
                self._mean.get_shape(), self._stdx.get_shape()
            )
            self._dynamic_batch_shape = tf.broadcast_dynamic_shape(
                tf.shape(self._mean), tf.shape(self._stdx)
            )

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

    def sample(self, sample_shape=()):
        with tf.name_scope(self.name_scope, values=[sample_shape]), \
                tf.name_scope('sample'):
            mean, stdx, stdx_is_log = self._mean, self._stdx, self._stdx_is_log
            dynamic_sample_shape = tf.concat(
                [sample_shape, self._dynamic_batch_shape],
                axis=0
            )
            dtype = mean.dtype.base_dtype
            samples = mean + (
                self.stddev *
                tf.random_normal(dynamic_sample_shape, dtype=dtype)
            )
            static_sample_shape = tf.TensorShape(
                tuple(None if not is_integer(i) else i for i in sample_shape))
            samples.set_shape(
                static_sample_shape.concatenate(self._static_batch_shape))
            return samples

    def log_prob(self, x):
        with tf.name_scope(self.name_scope, values=[x]), \
                tf.name_scope('log_prob'):
            c = -0.5 * np.log(2 * np.pi)
            return (
                c - self.logstd -
                0.5 * self.precision * tf.square(x - self.mean)
            )

    def prob(self, x):
        with tf.name_scope(self.name_scope, values=[x]), \
                tf.name_scope('prob'):
            return tf.exp(self.log_prob(x))

    def analytic_kld(self, other):
        if isinstance(other, Normal):
            with tf.name_scope(self.name_scope), tf.name_scope('normal_kld'):
                return 0.5 * (
                    self.var * other.precision +
                    tf.square(other.mean - self.mean) * other.precision +
                    other.logvar - self.logvar - 1
                )
        raise NotImplementedError()
