# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.distributions import Distribution
from tfsnippet.utils import (VarScopeObject,
                             is_integer,
                             instance_reuse,
                             open_variable_scope,
                             TensorArithmeticMixin)

__all__ = ['StochasticTensor']


class StochasticTensor(VarScopeObject, TensorArithmeticMixin):
    """Tensor-like object that represents a random variable in graph.

    Parameters
    ----------
    distribution : Distribution
        The distribution of this stochastic tensor.

    sample_num : int | tf.Tensor | tf.Variable
        Optional number of samples to take. (default None)

        If specified, even if 1, the sample(s) will occupy a new dimension
        before `batch_shape + value_shape` of `distrib`.
        Otherwise only one sample will be taken and it will not occupy
        a new dimension.

    observed : tf.Tensor
        If specified, random sampling will not take place.
        Instead this tensor will be regarded as sampled tensor.
        (default None)

        Note that if `sample_num` is specified, the shape of this
        tensor must be `[?] + batch_shape + value_shape`.
        And if `sample_num` is not specified, its shape must be
        `batch_shape + value_shape`.

    name, default_name : str
        Optional name or default name of this stochastic tensor.
    """

    def __init__(self, distribution, sample_num=None, observed=None,
                 name=None, default_name=None):
        super(StochasticTensor, self).__init__(
            name=name,
            default_name=default_name
        )

        with open_variable_scope(self.variable_scope), tf.name_scope('init'):
            if sample_num is not None:
                if is_integer(sample_num):
                    if sample_num < 1:
                        raise ValueError('`sample_num` must be at least 1.')
                    sample_shape = (sample_num,)
                    static_sample_shape = (sample_num,)
                else:
                    sample_num = tf.convert_to_tensor(sample_num)
                    sample_num_shape = sample_num.get_shape()
                    if not sample_num.dtype.is_integer or (
                                    sample_num_shape.ndims is not None and
                                    sample_num_shape.ndims != 0):
                        raise ValueError('`sample_num` must be an integer '
                                         'scalar.')
                    sample_shape = tf.stack([sample_num])
                    static_sample_shape = (None,)
            else:
                sample_shape = ()
                static_sample_shape = ()
            if observed is not None:
                observed = tf.convert_to_tensor(
                    observed,
                    dtype=distribution.dtype
                )

            self._distrib = distribution
            self._sample_num = sample_num
            self._sample_shape = sample_shape
            self._static_sample_shape = (
                tf.TensorShape(list(static_sample_shape)).
                    concatenate(distribution.static_batch_shape).
                    concatenate(distribution.static_value_shape)
            )
            self._observed_tensor = observed
            self._computed_tensor = observed    # type: tf.Tensor

    @property
    def distribution(self):
        """Get the distribution."""
        return self._distrib

    @property
    def dtype(self):
        """Get the data type of this stochastic tensor."""
        return self._distrib.dtype

    @property
    def sample_num(self):
        """Get the sample number."""
        return self._sample_num

    @property
    def is_observed(self):
        """Whether or not this stochastic tensor is observed?"""
        return self._observed_tensor is not None

    @property
    def observed_tensor(self):
        """Get the observed tensor, if specified."""
        return self._observed_tensor

    @property
    @instance_reuse
    def computed_tensor(self):
        """Get the observed or sampled tensor."""
        if self._computed_tensor is None:
            if self._observed_tensor is not None:
                self._computed_tensor = self._observed_tensor
            else:
                self._computed_tensor = self._distrib.sample(self._sample_shape)
        return self._computed_tensor

    @property
    def get_shape(self):
        """Get the static shape of this stochastic tensor."""
        return self._static_sample_shape

    def __repr__(self):
        return 'StochasticTensor(%r)' % (self.computed_tensor,)

    def __hash__(self):
        return hash(self.computed_tensor)

    def __getitem__(self, item):
        return self.computed_tensor[item]


def _stochastic_to_tensor(value, dtype=None, name=None, as_ref=False):
    if dtype and not dtype.is_compatible_with(value.dtype):
        raise ValueError('Incompatible type conversion requested to type '
                         '%s for variable of type %s' %
                         (dtype.name, value.dtype.name))
    if as_ref:
        raise ValueError('%r: Ref type not supported.' % value)
    return value.computed_tensor


tf.register_tensor_conversion_function(StochasticTensor, _stochastic_to_tensor)
