# -*- coding: utf-8 -*-
import numpy as np

from tfsnippet import config

__all__ = [
    'floatx', 'set_floatx', 'cast_to_floatx'
]


# check the configured value of floatx
_FLOATX = config['FLOATX']
if _FLOATX not in {'float16', 'float32', 'float64'}:
    raise ValueError('Invalid floatx config value: %r.' % (_FLOATX,))


def floatx():
    """Returns the default float type, as a string.

    Returns
    -------
    {'float16', 'float32', 'float64'}
        The default float type.
    """
    return _FLOATX


def set_floatx(floatx):
    """Sets the default float type.

    Parameters
    ----------
    floatx : {'float16', 'float32', or 'float64'}
        The default float type.
    """
    global _FLOATX
    if floatx not in {'float16', 'float32', 'float64'}:
        raise ValueError('Unknown floatx type: ' + str(floatx))
    _FLOATX = str(floatx)
    config['FLOATX'] = _FLOATX


def cast_to_floatx(x):
    """Cast a Numpy array to the default float type.

    Parameters
    ----------
    x : np.ndarray
        The numpy array to be casted.

    Returns
    -------
    np.ndarray
        The same Numpy array, cast to its new type.
    """
    return np.asarray(x, dtype=_FLOATX)
