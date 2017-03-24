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
    """

    def __init__(self, mean, stddev=None, logstd=None):
        stdx = stddev if stddev is not None else logstd
        if stdx is None:
            raise ValueError('One of `stddev`, `logstd` should be specified.')

        # check the shape of parameters
        self._mean = tf.convert_to_tensor(mean, dtype=floatx())
        self._stdx = tf.convert_to_tensor(stdx, dtype=floatx())
        if stddev is None:
            self._stdx_is_log = True
            self._stddev = tf.exp(self._stdx)
            self._logstd = self._stdx
        else:
            self._stdx_is_log = False
            self._stddev = self._stdx
            self._logstd = tf.log(self._stdx)
        try:
            tf.broadcast_static_shape(self._mean.get_shape(),
                                      self._stdx.get_shape())
        except ValueError:
            raise ValueError('`mean` and `stddev`/`logstd` should be '
                             'broadcastable to match each other (%r vs %r).' %
                             (self._mean.get_shape(), self._stdx.get_shape()))

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

    def sample(self, sample_shape=()):
        mean, stdx, stdx_is_log = self._mean, self._stdx, self._stdx_is_log
        dynamic_sample_shape = tf.concat(
            [sample_shape, self._dynamic_batch_shape],
            axis=0
        )
        dtype = mean.dtype.base_dtype
        samples = (
            mean +
            self.stddev * tf.random_normal(dynamic_sample_shape, dtype=dtype)
        )
        static_sample_shape = tf.TensorShape(
            tuple(None if not is_integer(i) else i for i in sample_shape))
        samples.set_shape(
            static_sample_shape.concatenate(self._static_batch_shape))
        return samples

    def log_prob(self, x):
        c = -0.5 * np.log(2 * np.pi)
        if self._stdx_is_log:
            precision = tf.exp(-2 * self._stdx)
        else:
            precision = 1. / tf.square(self._stdx)
        return c - self.logstd - 0.5 * precision * tf.square(x - self.mean)

    def prob(self, x):
        return tf.exp(self.log_prob(x))
