# -*- coding: utf-8 -*-
import six
import numpy as np
import tensorflow as tf

from mlcomp.utils import camel_to_underscore

__all__ = [
    'camel_to_underscore', 'is_integer', 'is_deterministic_shape',
]


def is_integer(x):
    """Test whether or not `x` is a Python integer or a Numpy integer."""
    return isinstance(x, six.integer_types + (np.integer,))


def is_deterministic_shape(x):
    """Test whether or not shape `x` is deterministic.

    A deterministic shape should be a tuple or a list of integers.
    Otherwise if `x` is an instance of `tf.TensorShape`, a `tf.Tensor`,
    or a tuple / list which contains integers but also tf.Tensor as elements,
    it should be a non-deterministic shape.

    If `x` is a tuple / list which contains unexpected types of elements,
    or if it is a tuple / list / tf.Tensor / tf.TensorShape, a TypeError
    will be raised.
    """
    if isinstance(x, (tuple, list)):
        deterministic = True
        for i in x:
            if i is None or isinstance(i, (tf.Tensor, tf.Variable)):
                return False
            elif not is_integer(i):
                deterministic = False
                break
        if deterministic:
            return True
    elif isinstance(x, tf.TensorShape):
        return x.is_fully_defined()
    elif isinstance(x, (tf.Tensor, tf.Variable)):
        return False
    raise TypeError('%r is not a shape object.' % (x,))
