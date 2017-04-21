# -*- coding: utf-8 -*-
from contextlib import contextmanager

import six
import tensorflow as tf
from tensorflow.python.ops import variable_scope as variable_scope_ops

__all__ = [
    'open_variable_scope'
]


@contextmanager
def open_variable_scope(name_or_scope,
                        reuse=None,
                        initializer=None,
                        regularizer=None,
                        caching_device=None,
                        partitioner=None,
                        custom_getter=None,
                        dtype=tf.float32):
    """Open or re-open a variable scope, ensuring to open `original_name_scope`.
    
    When using `tf.variable_scope` to re-open an existing variable scope, the
    original name scope will not be open.  Instead a different name scope will
    be opened under the current name scope.
    
    This method thus fixes this issue, by ensuring to open the original name
    scope if a variable scope is re-opened.
    
    Parameters
    ----------
    name_or_scope : str | tf.VariableScope
        The name of the scope, or the variable scope object.
         
    reuse : None | bool
        Whether or not to reuse the variables in opened scope?
        
    initializer, regularizer, caching_device, partitioner, custom_getter, dtype
        Other parameters for opening the variable scope.

    Yields
    ------
    tf.VariableScope
        The opened variable scope.
    """
    if isinstance(name_or_scope, six.string_types):
        old = tf.get_variable_scope()
        if old and old.original_name_scope:
            old_name_scope = old.original_name_scope
            if not old_name_scope.endswith('/'):
                old_name_scope += '/'
        else:
            old_name_scope = ''
        old_name_scope += name_or_scope
        if not old_name_scope.endswith('/'):
            old_name_scope += '/'
    else:
        old_name_scope = name_or_scope.original_name_scope

    with variable_scope_ops._pure_variable_scope(
            name_or_scope,
            reuse=reuse,
            initializer=initializer,
            regularizer=regularizer,
            caching_device=caching_device,
            partitioner=partitioner,
            custom_getter=custom_getter,
            old_name_scope=old_name_scope,
            dtype=dtype) as pure_variable_scope:
        name_scope = pure_variable_scope.original_name_scope
        if not name_scope.endswith('/'):
            name_scope += '/'
        with tf.name_scope(name_scope):
            yield pure_variable_scope
