# -*- coding: utf-8 -*-
import six
import numpy as np
import tensorflow as tf

__all__ = [
    'is_integer', 'is_float', 'is_dynamic_tensor_like',
    'convert_to_tensor_if_dynamic',
    'get_preferred_tensor_dtype',
]


def is_integer(x):
    """Test whether or not `x` is a Python or Numpy integer."""
    if isinstance(x, bool):
        return False
    return isinstance(x, __INTEGER_TYPES)

__INTEGER_TYPES = (
    six.integer_types +
    (np.integer, np.int, np.uint,
     np.int8, np.int16, np.int32, np.int64,
     np.uint8, np.uint16, np.uint32, np.uint64)
)


def is_float(x):
    """Test whether or not `x` is a Python or Numpy floating number."""
    return isinstance(x, __FLOATING_TYPES)

__FLOATING_TYPES = (
    float,
    np.float,
    np.float16, np.float32, np.float64, np.float128,
)


def is_dynamic_tensor_like(x):
    """Check whether or not `x` should be converted by `tf.convert_to_tensor`.

    Parameters
    ----------
    x
        The object to be checked.
    """
    from tfsnippet.bayes import StochasticTensor
    return isinstance(x, (tf.Tensor, tf.Variable, StochasticTensor))


def convert_to_tensor_if_dynamic(x, name=None):
    """Convert `x` to tensor if it is dynamic.

    Parameters
    ----------
    x
        The object to be converted.

    name : str
        Optional name of this operation.
    """
    if is_dynamic_tensor_like(x):
        return tf.convert_to_tensor(x, name=name)
    return x


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
    if hasattr(x, 'dtype'):
        dtype = x.dtype
        return tf.as_dtype(dtype).base_dtype
    else:
        return tf.as_dtype(np.array([x]).dtype)
