# -*- coding: utf-8 -*-
import six
import numpy as np
import tensorflow as tf

__all__ = [
    'is_integer', 'is_float', 'is_dynamic_tensor_like',
    'convert_to_tensor_if_dynamic', 'get_preferred_tensor_dtype',
    'MetricAccumulator', 'humanize_duration',
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


class MetricAccumulator(object):
    """Accumulator to compute the average of certain metric.

    Parameters
    ----------
    initial_value : float
        The initial value of this accumulator (default 0.)

    initial_weight : float
        The initial weight of this accumulator (default 0.)
    """

    def __init__(self, initial_value=0., initial_weight=0.):
        if initial_weight < 0:
            raise ValueError('`initial_weight` must be greater or equal to 0.')
        self._initial_value = float(initial_value)
        self._initial_weight = float(initial_weight)
        self._value = initial_value
        self._weight = initial_weight
        self._counter = 0

    @property
    def weight(self):
        """Get the weight of accumulated values."""
        return self._weight

    @property
    def value(self):
        """Get the average value, or None if no value has been collected."""
        return self._value

    @property
    def has_value(self):
        """Whether or not any value has been collected?"""
        return self._counter > 0

    @property
    def counter(self):
        """Get the counter of added values."""
        return self._counter

    def reset(self):
        """Reset the accumulator to initial state."""
        self._value = self._initial_value
        self._weight = self._initial_weight
        self._counter = 0

    def add(self, value, weight=1.):
        """Add a value to this accumulator.

        Parameters
        ----------
        value : float
            The value to be collected.

        weight : float
            Optional weight of this value (default 1.)
        """
        self._weight += weight
        self._value += (value - self._value) * (weight / self._weight)
        self._counter += 1


def humanize_duration(seconds):
    """Format specified time duration into human readable text.

    Parameters
    ----------
    seconds : int | float
        Number of seconds, as the time duration.

    Returns
    -------
    str
        The formatted time duration.
    """
    if seconds < 0:
        seconds = -seconds
        suffix = ' ago'
    else:
        suffix = ''

    pieces = []
    for uvalue, uname in [(86400, 'day'),
                          (3600, 'hr'),
                          (60, 'min')]:
        if seconds >= uvalue:
            val = int(seconds // uvalue)
            if val > 0:
                if val > 1:
                    uname += 's'
                pieces.append('%d %s' % (val, uname))
            seconds %= uvalue
    if seconds > np.finfo(np.float64).eps:
        pieces.append('%.4g sec%s' % (seconds, 's' if seconds > 1 else ''))
    elif not pieces:
        pieces.append('0 sec')

    return ' '.join(pieces) + suffix
