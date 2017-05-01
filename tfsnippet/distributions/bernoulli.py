# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.utils import (get_preferred_tensor_dtype, open_variable_scope,
                             is_deterministic_shape, ReshapeHelper,
                             maybe_explicit_broadcast)
from .base import Distribution

__all__ = ['Bernoulli']


class Bernoulli(Distribution):
    """Bernoulli distribution.

    Parameters
    ----------
    logits : tf.Tensor | np.ndarray | float
        A float tensor, which is the log-odds of probabilities of being 1.

        The relationship between Bernoulli `p` and `logits` are:
        
            .. math::
                \\begin{aligned}
                    \\text{logits} &= \\log \\frac{p}{1-p} \\\\
                                 p &= \\frac{1}{1 + \\exp(-\\text{logits})} 
                \\end{aligned}
                
        Note that the range of `logits` is :math:`(-\\infty, \\infty)`.

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

    def __init__(self, logits, dtype=None, group_event_ndims=None,
                 name=None, default_name=None):
        param_dtype = get_preferred_tensor_dtype(logits)
        if not param_dtype.is_floating:
            raise TypeError('Bernoulli distribution parameters must be float '
                            'numbers.')
        if dtype is None:
            dtype = tf.int32
        else:
            dtype = tf.as_dtype(dtype)
            if not dtype.is_integer:
                raise TypeError('`dtype` is expected to be an integer type, '
                                'but got %r.' % (dtype,))

        super(Bernoulli, self).__init__(
            group_event_ndims=group_event_ndims,
            name=name,
            default_name=default_name,
        )

        with open_variable_scope(self.variable_scope), tf.name_scope('init'):
            # obtain parameter tensors
            logits = tf.convert_to_tensor(logits, dtype=param_dtype)
            logits_shape = logits.get_shape()

            # derive the shape and data types of parameters
            self._logits = logits
            self._static_batch_shape = logits_shape
            if is_deterministic_shape(logits_shape):
                self._dynamic_batch_shape = \
                    tf.convert_to_tensor(logits_shape.as_list(), dtype=tf.int32)
            else:
                self._dynamic_batch_shape = tf.shape(logits)

            # derive various distribution attributes
            self._p = tf.nn.sigmoid(logits, 'p')
            self._one_minus_p = tf.nn.sigmoid(-logits, 'one_minus_p')

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
    def logits(self):
        """Get the logits of Bernoulli distribution."""
        return self._logits

    @property
    def mean(self):
        """Get the mean of Bernoulli distribution."""
        return self._p

    @property
    def p(self):
        """Get the probabilities of being 1."""
        return self._p

    @property
    def p_take_zero(self):
        """Get the probability of being 0."""
        return self._one_minus_p

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
            uniform_samples = tf.random_uniform(
                dynamic_shape, minval=0., maxval=1., dtype=self.param_dtype)
            samples = tf.cast(
                tf.less(uniform_samples, self.p),
                dtype=self.dtype
            )
            samples.set_shape(static_shape)
            return samples

    def _log_prob(self, x):
        x = tf.cast(x, dtype=self.param_dtype)
        logits = self.logits
        x, logits = maybe_explicit_broadcast(x, logits)
        return -tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits)

    def analytic_kld(self, other, name=None):
        if isinstance(other, Bernoulli):
            with tf.name_scope(name, default_name='analytic_kld'):
                p_1, p_0 = self.p, self.p_take_zero
                delta_1 = (
                    tf.nn.softplus(-other.logits) -
                    tf.nn.softplus(-self.logits)
                )
                delta_0 = (
                    tf.nn.softplus(other.logits) -
                    tf.nn.softplus(self.logits)
                )
                return p_1 * delta_1 + p_0 * delta_0
        raise NotImplementedError()
