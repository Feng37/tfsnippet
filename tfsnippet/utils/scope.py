# -*- coding: utf-8 -*-
from contextlib import contextmanager

import six
import tensorflow as tf
from tensorflow.python.ops import variable_scope as variable_scope_ops

from .imported import camel_to_underscore

__all__ = [
    'open_variable_scope', 'ScopedObject',
    'get_variables_as_dict',
]


@contextmanager
def open_variable_scope(name_or_scope,
                        pure_variable_scope=False,
                        unique_name_scope=False,
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
    scope if a variable scope is re-opened, if `keep_name_scope` is required.
    
    Notes
    -----
    This method does not support `default_name`, which is used to suggest
    a name template for choosing a unique variable scope name, used in
    `tf.variable_scope`.  It opens exactly `name_or_scope`.
    
    Parameters
    ----------
    name_or_scope : str | tf.VariableScope
        The name of the scope, or the variable scope object.
        
    pure_variable_scope : bool
        Whether or not to open a pure variable scope?
        
        If set to True, a pure variable scope will be opened without
        changing the current name scope. (default is False)

    unique_name_scope : bool
        Whether or not to open a unique name scope?
        This argument will be ignored if `pure_variable_scope` is True.

        If set to True, a unique name scope will be opened.
        Otherwise the original name scope of the variable scope will be
        reopened. (default is False)
                  
    reuse : None | bool
        Whether or not to reuse the variables in opened scope?
        
    initializer, regularizer, caching_device, partitioner, custom_getter, dtype
        Other parameters for opening the variable scope.

    Yields
    ------
    tf.VariableScope
        The opened variable scope.
    """
    graph = tf.get_default_graph()

    if isinstance(name_or_scope, tf.VariableScope):
        old_name_scope = name_or_scope.original_name_scope
    elif isinstance(name_or_scope, six.string_types):
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
        raise TypeError(
            '`name_or_scope` is neither a tf.VariableScope nor a string.')

    if unique_name_scope and not old_name_scope.strip('/'):
        raise ValueError('`unique_name_scope` can be set to True only if '
                         'the variable scope to open is not the root scope.')

    with variable_scope_ops._pure_variable_scope(
            name_or_scope,
            reuse=reuse,
            initializer=initializer,
            regularizer=regularizer,
            caching_device=caching_device,
            partitioner=partitioner,
            custom_getter=custom_getter,
            old_name_scope=old_name_scope,
            dtype=dtype) as vs:
        if not pure_variable_scope:
            name_scope = vs.original_name_scope.rstrip('/')

            # query whether or not the full name scope has been used
            # if not, we must open the name scope by unique routine, otherwise
            # the name scope will not be marked as opened
            is_first_open = name_scope and (
                graph.unique_name(name_scope, mark_as_used=False) == name_scope
            )
            if is_first_open:
                unique_name_scope = True

            # open the parent name scope, then open the child by relative name,
            # so that the child scope will be unique.
            if unique_name_scope:
                scope_segments = name_scope.rsplit('/', 1)
                if len(scope_segments) > 1:
                    parent_scope = scope_segments[0] + '/'
                    child_scope = scope_segments[1]
                else:
                    parent_scope = ''
                    child_scope = scope_segments[0]

                with tf.name_scope(parent_scope):
                    with tf.name_scope(child_scope):
                        yield vs

            # otherwise open the full name scope directly
            else:
                if name_scope:
                    name_scope += '/'
                with tf.name_scope(name_scope):
                    yield vs
        else:
            yield vs


class ScopedObject(object):
    """Base class for all objects owns a variable scope.
    
    Such an object is compatible with the use patterns of `instance_reuse`.

    Parameters
    ----------
    name : str
        Name of this object.
        
        Note that if `name` is specified, the variable scope of constructed
        object will take exactly this name, even if another object has already
        taken such name.  That is to say, these two objects will share the same
        variable scope.
        
    default_name : str
        Default name of this object.
        
        When `name` is not specified, the object will obtain a variable scope
        with unique name, according to the name suggested by `default_name`.
        If `default_name` is also not specified, it will use the underscored
        class name as the default name.
    """

    def __init__(self, name=None, default_name=None):
        self.name = name

        # get the TensorFlow variable scope
        if not default_name:
            default_name = camel_to_underscore(self.__class__.__name__)
        with tf.variable_scope(name, default_name=default_name) as vs:
            self._variable_scope = vs       # type: tf.VariableScope

    @property
    def variable_scope(self):
        """Get the variable scope of this object."""
        return self._variable_scope


def get_variables_as_dict(scope=None, collection=tf.GraphKeys.GLOBAL_VARIABLES):
    """Get all the variables as dict.

    Parameters
    ----------
    scope : str | tf.VariableScope
        If specified, will get the variables only within this scope.
        The name of the variable scope, or the variable scope object.

    collection : str
        Get the variables only belong this collection.
        (default is tf.GraphKeys.GLOBAL_VARIABLES)

    Returns
    -------
    dict[str, tf.Variable]
        Dict which maps from names to variable instances.

        The name will be the full names of variables if `scope` is not
        specified, or the relative names within the `scope`.
        By "relative name" we mean the variable names without the common
        prefix of the scope name.
    """
    # get the common prefix to be stripped
    if isinstance(scope, tf.VariableScope):
        scope_name = scope.name
    else:
        scope_name = scope
    if scope_name and not scope_name.endswith('/'):
        scope_name += '/'
    scope_name_len = len(scope_name) if scope_name else 0

    # get the variables and strip the prefix
    variables = tf.get_collection(collection, scope_name)
    return {
        var.name[scope_name_len:].rsplit(':', 1)[0]: var
        for var in variables
    }
