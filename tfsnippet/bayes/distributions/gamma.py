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

    def __init__(self, alpha, beta, group_event_ndims=None,
                 check_numerics=False, name=None, default_name=None):
        # check the arguments
        dtype = get_preferred_tensor_dtype(alpha)
        if not dtype.is_floating:
            raise TypeError('Gamma distribution parameters must be float '
                            'numbers.')

        super(Gamma, self).__init__(group_event_ndims=group_event_ndims,
                                    check_numerics=check_numerics,
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
    def enum_value_count(self):
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

    def _sample_n(self, n):
        return tf.random_gamma(
            tf.stack([n]), alpha=self.alpha, beta=self.beta, dtype=self.dtype)

    def _log_prob(self, x):
        x = tf.convert_to_tensor(x, dtype=self.param_dtype)
        log_beta = self._do_check_numerics(tf.log(self.beta), 'log(beta)')
        log_x = self._do_check_numerics(tf.log(x), 'log(x)')
        lgamma_alpha = self._do_check_numerics(
            tf.lgamma(self.alpha), 'lgamma(alpha)'
        )
        return (
            self.alpha * log_beta + (self.alpha - 1) * log_x -
            self.beta * x - lgamma_alpha
        )

    def _analytic_kld(self, other):
        if isinstance(other, Gamma):
            self_digamma_alpha = self._do_check_numerics(
                tf.digamma(self.alpha), 'digamma(alpha)'
            )
            self_lgamma_alpha = self._do_check_numerics(
                tf.lgamma(self.alpha), 'lgamma(alpha)'
            )
            self_log_beta = self._do_check_numerics(
                tf.log(self.beta), 'log(beta)'
            )
            other_lgamma_alpha = other._do_check_numerics(
                tf.lgamma(other.alpha), 'lgamma(alpha)'
            )
            other_log_beta = other._do_check_numerics(
                tf.log(other.beta), 'log(beta)'
            )

            return (
                ((self.alpha - other.alpha) * self_digamma_alpha) -
                self_lgamma_alpha + other_lgamma_alpha +
                other.alpha * (self_log_beta - other_log_beta) +
                self.alpha * (other.beta / self.beta - 1.)
            )
