# -*- coding: utf-8 -*-
import tensorflow as tf

from distutils.version import StrictVersion

__all__ = ['is_tensorflow_version_higher_or_equal']


def is_tensorflow_version_higher_or_equal(version):
    """Check whether the version of TensorFlow is higher than `version`.
    
    Parameters
    ----------
    version : str
        TensorFlow version as string.
        
    Returns
    -------
    bool
        True if higher or equal to, False if not.
    """
    return StrictVersion(version) <= StrictVersion(tf.__version__)
