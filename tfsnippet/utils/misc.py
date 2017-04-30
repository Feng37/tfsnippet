# -*- coding: utf-8 -*-
import six
import numpy as np
import tensorflow as tf

__all__ = [
    'is_integer', 'get_preferred_tensor_dtype',
]


def is_integer(x):
    """Test whether or not `x` is a Python integer or a Numpy integer."""
    if isinstance(x, bool):
        return False
    return isinstance(x, __INTEGER_TYPES)

__INTEGER_TYPES = (
    six.integer_types +
    (np.integer, np.int, np.uint,
     np.int8, np.int16, np.int32, np.int64,
     np.uint8, np.uint16, np.uint32, np.uint64)
)


def get_preferred_tensor_dtype(x):
    """Get the preferred dtype for specified tensor `x`.
    
    Parameters
    ----------
    x : tf.Tensor | tf.Variable | np.ndarray | int | float
        The tensor or scalar whose dtype should be inferred.
        
        If `x` is a TensorFlow tensor or variable, returns its base dtype.
        If `x` is a NumPy array, returns the dtype of array.
        If `x` is of other types, the corresponding dtype for a numpy array
        consisting only `x` will be used as the dtype.

    Returns
    -------
    tf.DType
        The data type for specified tensor `x`.
    """
    if isinstance(x, (tf.Tensor, tf.Variable)):
        return x.dtype.base_dtype
    elif isinstance(x, np.ndarray):
        return tf.as_dtype(x.dtype)
    else:
        return tf.as_dtype(np.array([x]).dtype)
