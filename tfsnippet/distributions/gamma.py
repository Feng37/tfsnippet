# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.utils import get_preferred_tensor_dtype, reopen_variable_scope
from .base import Distribution

__all__ = ['Gamma']


class Gamma(Distribution):
    """Univariate gamma distribution.

    Parameters
    ----------
    alpha : tf.Tensor | np.ndarray | float
        The shape parameter of the Gamma distribution.
        Should be positive and broadcastable to match `beta`.

        Note that the range of `alpha` is :math:`(0, \\infty)`.

    beta : tf.Tensor | np.ndarray | float
        The inverse scale parameter of the Gamma distribution.
        Should be positive and broadcastable to match `alpha`.
        Also, the data type of `beta` will be casted to that of `alpha`.

        Note that the range of `beta` is :math:`(0, \\infty)`.

    group_event_ndims : int
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

    name : str
        Name of this normal distribution.

    default_name : str
        Default name of this normal distribution.
    """

    def __init__(self, alpha, beta, group_event_ndims=None,
                 name=None, default_name=None):
        # check the arguments
        dtype = get_preferred_tensor_dtype(alpha)
        if not dtype.is_floating:
            raise TypeError('Gamma distribution parameters must be float '
                            'numbers.')

        super(Gamma, self).__init__(group_event_ndims=group_event_ndims,
                                    name=name,
                                    default_name=default_name)

        with reopen_variable_scope(self.variable_scope):
            with tf.name_scope('init'):
                # obtain parameter tensors
                self._alpha = alpha = tf.convert_to_tensor(alpha, dtype=dtype)
                self._beta = beta = tf.convert_to_tensor(beta, dtype=dtype)

                # check the shape and data types of parameters
                try:
                    self._static_batch_shape = tf.broadcast_static_shape(
                        alpha.get_shape(),
                        beta.get_shape()
                    )
                except ValueError:
                    raise ValueError(
                        '`alpha` and `beta` should be '
                        'broadcastable to match each other (%r vs %r).' %
                        (alpha.get_shape(), beta.get_shape())
                    )
                self._dynamic_batch_shape = tf.broadcast_dynamic_shape(
                    tf.shape(alpha), tf.shape(beta)
                )

    @property
    def dtype(self):
        return self._alpha.dtype.base_dtype

    @property
    def param_dtype(self):
        return self._alpha.dtype.base_dtype

    @property
    def is_continuous(self):
        return True

    @property
    def is_reparameterized(self):
        return False

    @property
    def is_enumerable(self):
        return False

    @property
    def n_enum_values(self):
        return None

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
    def alpha(self):
        """Get the shape parameter of the Gamma distribution."""
        return self._alpha

    @property
    def beta(self):
        """Get the inverse scale parameter of the Gamma distribution."""
        return self._beta

    def _sample(self, sample_shape=()):
        return tf.random_gamma(
            tf.convert_to_tensor(sample_shape, dtype=tf.int32),
            alpha=self.alpha, beta=self.beta, dtype=self.dtype)

    def _enum_sample(self):
        raise RuntimeError('Gamma distribution is not enumerable.')

    def _log_prob(self, x):
        x = tf.convert_to_tensor(x, dtype=self.param_dtype)
        return (
            self.alpha * tf.log(self.beta) + (self.alpha - 1) * tf.log(x) -
            self.beta * x - tf.lgamma(self.alpha)
        )

    def _analytic_kld(self, other):
        if isinstance(other, Gamma):
            return (
                ((self.alpha - other.alpha) * tf.digamma(self.alpha)) -
                tf.lgamma(self.alpha) + tf.lgamma(other.alpha) +
                other.alpha * (tf.log(self.beta) - tf.log(other.beta)) +
                self.alpha * (other.beta / self.beta - 1.)
            )
