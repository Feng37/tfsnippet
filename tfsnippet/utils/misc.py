# -*- coding: utf-8 -*-
import six
import numpy as np

__all__ = [
    'is_integer',
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
