# -*- coding: utf-8 -*-
import tensorflow as tf

__all__ = ['TensorArithmeticMixin']


class TensorArithmeticMixin(object):
    """Mixin class for implementing tensor arithmetic operations.

    The derived class must support `tf.convert_to_tensor`, in order to
    inherit from this mixin class.
    """

    # overloading arithmetic operations
    def __abs__(self):
        return tf.abs(self)

    def __neg__(self):
        return tf.negative(self)

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

    def __rdiv__(self, other):
        return tf.div(other, self)

    def __truediv__(self, other):
        return tf.truediv(self, other)

    def __rtruediv__(self, other):
        return tf.truediv(other, self)

    def __floordiv__(self, other):
        return tf.floordiv(self, other)

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

    def __rand__(self, other):
        return tf.logical_and(other, self)

    def __or__(self, other):
        return tf.logical_or(self, other)

    def __ror__(self, other):
        return tf.logical_or(other, self)

    def __xor__(self, other):
        return tf.logical_xor(self, other)

    def __rxor__(self, other):
        return tf.logical_xor(other, self)

    # boolean operations
    def __lt__(self, other):
        return tf.less(self, other)

    def __le__(self, other):
        return tf.less_equal(self, other)

    def __gt__(self, other):
        return tf.greater(self, other)

    def __ge__(self, other):
        return tf.greater_equal(self, other)

    # slicing and indexing
    def __getitem__(self, item):
        return (tf.convert_to_tensor(self))[item]
