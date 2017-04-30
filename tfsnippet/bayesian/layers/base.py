# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.distributions import Distribution
from tfsnippet.utils import VarScopeObject, is_integer

__all__ = ['StochasticTensor']


class StochasticTensor(VarScopeObject):
    """Tensor-like object that represents a random variable in graph.

    Parameters
    ----------
    distribution : Distribution
        The distribution of this stochastic tensor.

    sample_num : int | tf.Tensor
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
        if sample_num is not None:
            if is_integer(sample_num):
                if sample_num < 1:
                    raise ValueError('`sample_num` must be at least 1.')
                sample_shape = (sample_num,)
            elif isinstance(sample_num, (tf.Tensor, tf.Variable)):
                sample_num_shape = sample_num.get_shape()
                if not sample_num.dtype.is_integer or (
                                sample_num_shape.ndims is not None and
                                sample_num_shape.ndims != 0):
                    raise ValueError('`sample_num` must be an integer scalar.')
                sample_shape = tf.stack([sample_num])
            else:
                raise TypeError('`sample_num` is expected to be an integer '
                                'scalar, but got %r.' % (sample_num,))
        else:
            sample_shape = ()

        super(StochasticTensor, self).__init__(
            name=name,
            default_name=default_name
        )
        observed = tf.convert_to_tensor(
            observed,
            dtype=self.distribution.dtype
        )

        self._distrib = distribution
        self._sample_num = sample_num
        self._sample_shape = sample_shape
        self._observed_tensor = observed
        self._sampled_tensor = None  # type: tf.Tensor

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
    def tensor(self):
        """Get the observed or sampled tensor."""
        if self._sampled_tensor is None:
            if self._observed_tensor is not None:
                self._sampled_tensor = self._observed_tensor
            else:
                self._sampled_tensor = self._distrib.sample(self._sample_shape)
        return self._sampled_tensor

    # overloading arithmetic operations
    def __abs__(self):
        return tf.abs(self)

    def __neg__(self):
        return tf.negative(self)

    def __pos__(self):
        return self

    def __add__(self, other):
        return tf.add(self, other)

    def __radd__(self, other):
        return tf.add(other, self)

    def __sub__(self, other):
        return tf.subtract(self, other)

    def __rsub__(self, other):
        return tf.subtract(other, self)

    def __mul__(self, other):
        return tf.multiply(self, other)

    def __rmul__(self, other):
        return tf.multiply(other, self)

    def __div__(self, other):
        return tf.div(self, other)

    def __truediv__(self, other):
        return tf.truediv(self, other)

    def __floordiv__(self, other):
        return tf.floordiv(self, other)

    def __rdiv__(self, other):
        return tf.div(other, self)

    def __rtruediv__(self, other):
        return tf.truediv(other, self)

    def __rfloordiv__(self, other):
        return tf.floordiv(other, self)

    def __mod__(self, other):
        return tf.mod(self, other)

    def __rmod__(self, other):
        return tf.mod(other, self)

    def __pow__(self, other):
        return tf.pow(self, other)

    def __rpow__(self, other):
        return tf.pow(other, self)

    # logical operations
    def __invert__(self):
        return tf.logical_not(self)

    def __and__(self, other):
        return tf.logical_and(self, other)

    def __or__(self, other):
        return tf.logical_or(self, other)

    def __xor__(self, other):
        return tf.logical_xor(self, other)

    # boolean operations
    def __lt__(self, other):
        return tf.less(self, other)

    def __le__(self, other):
        return tf.less_equal(self, other)

    def __gt__(self, other):
        return tf.greater(self, other)

    def __ge__(self, other):
        return tf.greater_equal(self, other)

    def __eq__(self, other):
        return tf.equal(self, other)

    def __ne__(self, other):
        return tf.not_equal(self, other)

    def __hash__(self):
        return hash(self.tensor)

    @staticmethod
    def _to_tensor(value, dtype=None, name=None, as_ref=False):
        if dtype and not dtype.is_compatible_with(value.dtype):
            raise ValueError("Incompatible type conversion requested to type "
                             "'{}' for variable of type '{}'".
                             format(dtype.name, value.dtype.name))
        if as_ref:
            raise ValueError("{}: Ref type not supported.".format(value))
        return value.tensor


tf.register_tensor_conversion_function(
    StochasticTensor, StochasticTensor._to_tensor)
