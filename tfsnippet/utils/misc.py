# -*- coding: utf-8 -*-
import math
import re
import threading
from itertools import chain

import numpy as np
import six
import tensorflow as tf

__all__ = [
    'NOT_SPECIFIED',
    'is_integer', 'is_float', 'is_dynamic_tensor_like',
    'convert_to_tensor_if_dynamic', 'get_preferred_tensor_dtype',
    'MetricAccumulator', 'humanize_duration',
    'unique', 'AutoReprObject', 'ContextStack', 'camel_to_underscore',
]


class NotSpecified(object):
    """Type for `NOT_SPECIFIED` constant."""

NOT_SPECIFIED = NotSpecified()


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
    np.float16, np.float32, np.float64,
)

for _extra_float_type in ('float8', 'float128', 'float256'):
    if hasattr(np, _extra_float_type):
        __FLOATING_TYPES = __FLOATING_TYPES + (getattr(np, _extra_float_type),)


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
    """Accumulator to compute the statistics of certain metric."""

    def __init__(self):
        self._mean = 0.     # E[X]
        self._square = 0.   # E[X^2]
        self._weight = 0.
        self._counter = 0

    @property
    def weight(self):
        """Get the weight of accumulated values."""
        return self._weight

    @property
    def mean(self):
        """Get the average value, i.e., E[X]."""
        return self._mean

    @property
    def square(self):
        """Get E[X^2]."""
        return self._square

    @property
    def var(self):
        """Get the variance, i.e., E[X^2] - (E[X])^2."""
        return self._square - self._mean ** 2

    @property
    def stddev(self):
        """Get the standard derivation, i.e., sqrt(var)."""
        return math.sqrt(self.var)

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
        self._mean = 0.
        self._square = 0.
        self._weight = 0.
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
        discount = weight / self._weight
        self._mean += (value - self._mean) * discount
        self._square += (value ** 2 - self._square) * discount
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


def unique(iterable):
    """Uniquify elements to construct a new list.

    Parameters
    ----------
    iterable : collections.Iterable[any]
        The collection to be uniquified.

    Returns
    -------
    list[any]
        Unique elements as a list, whose orders are preserved.
    """
    def small():
        ret = []
        for e in iterable:
            if e not in ret:
                ret.append(e)
        return ret

    def large():
        ret = []
        memo = set()
        for e in iterable:
            if e not in memo:
                memo.add(e)
                ret.append(e)
        return ret

    if hasattr(iterable, '__len__'):
        if len(iterable) < 1000:
            return small()
    return large()


class AutoReprObject(object):
    """Base class for objects with automatic repr implementation.

    The automatic repr implementation will demonstrate all the attributes
    in repr function.  To control the order of these attributes, one may
    provide an attribute list in `__repr_attributes__` class variable.

    Note that by default, this class only formats instance attributes,
    which should appear in `__dict__` of an object. Class attributes
    are not included, unless listed in `__repr_attributes__`.
    """

    __repr_attributes__ = None

    def __repr__(self):
        repr_attrs = getattr(self, '__repr_attributes__') or ()
        repr_attrs = unique(chain(repr_attrs, sorted(self.__dict__)))
        pieces = ','.join(
            '%s=%s' % (k, repr(v))
            for k, v in ((k, getattr(self, k)) for k in repr_attrs) if v
        )
        return '%s(%s)' % (self.__class__.__name__, pieces)


class ContextStack(object):
    """Thread-local context stack for general purpose.

    Parameters
    ----------
    initial_factory : () -> any
        If specified, fill the context stack with an initial object
        generated by this factory.
    """

    _STORAGE_STACK_KEY = '_context_stack'

    def __init__(self, initial_factory=None):
        self._storage = threading.local()
        self._initial_factory = initial_factory

    @property
    def items(self):
        if not hasattr(self._storage, self._STORAGE_STACK_KEY):
            new_stack = []
            if self._initial_factory is not None:
                new_stack.append(self._initial_factory())
            setattr(self._storage, self._STORAGE_STACK_KEY, new_stack)
        return getattr(self._storage, self._STORAGE_STACK_KEY)

    def push(self, context):
        self.items.append(context)

    def pop(self):
        self.items.pop()

    def top(self):
        items = self.items
        if items:
            return items[-1]


def camel_to_underscore(name):
    """Convert a camel-case name to underscore."""
    s1 = re.sub(CAMEL_TO_UNDERSCORE_S1, r'\1_\2', name)
    return re.sub(CAMEL_TO_UNDERSCORE_S2, r'\1_\2', s1).lower()

CAMEL_TO_UNDERSCORE_S1 = re.compile('([^_])([A-Z][a-z]+)')
CAMEL_TO_UNDERSCORE_S2 = re.compile('([a-z0-9])([A-Z])')
