# -*- coding: utf-8 -*-
import warnings
from contextlib import contextmanager

import six
import tensorflow as tf
from tensorflow.python.ops import variable_scope as variable_scope_ops

from .misc import camel_to_underscore

__all__ = [
    'get_variables_as_dict',
    'reopen_variable_scope', 'root_variable_scope',
    'lagacy_default_name_arg', 'VarScopeObject',
]


@contextmanager
def reopen_variable_scope(var_scope,
                          reuse=None,
                          initializer=None,
                          regularizer=None,
                          caching_device=None,
                          partitioner=None,
                          custom_getter=None,
                          dtype=tf.float32):
    """Reopen the specified `var_scope` and its `original_name_scope`.

    `tf.variable_scope` will not open the original name scope, even if a
    stored `tf.VariableScope` instance is specified.  This method thus
    allows to open exactly the same name scope as the original one.

    Parameters
    ----------
    var_scope : tf.VariableScope
        The variable scope instance.

    reuse : None | bool
        Whether or not to reuse the variables in opened scope?

    initializer, regularizer, caching_device, partitioner, custom_getter, dtype
        Other parameters for opening the variable scope.
    """
    if not isinstance(var_scope, tf.VariableScope):
        raise TypeError('`var_scope` is expected to be an instance of '
                        '`tf.VariableScope`.')
    old_name_scope = var_scope.original_name_scope

    with variable_scope_ops._pure_variable_scope(
            var_scope,
            reuse=reuse,
            initializer=initializer,
            regularizer=regularizer,
            caching_device=caching_device,
            partitioner=partitioner,
            custom_getter=custom_getter,
            old_name_scope=old_name_scope,
            dtype=dtype) as vs:
        name_scope = old_name_scope
        if name_scope and not name_scope.endswith('/'):
            name_scope += '/'

        with tf.name_scope(name_scope):
            yield vs


@contextmanager
def root_variable_scope(reuse=None,
                        initializer=None,
                        regularizer=None,
                        caching_device=None,
                        partitioner=None,
                        custom_getter=None,
                        dtype=tf.float32):
    """Open the root name and variable scope.

    Parameters
    ----------
    reuse : None | bool
        Whether or not to reuse the variables in opened scope?

    initializer, regularizer, caching_device, partitioner, custom_getter, dtype
        Other parameters for opening the variable scope.
    """
    # `tf.variable_scope` does not support opening the root variable scope
    # from empty name.  It always prepend the name of current variable scope
    # to the front of opened variable scope.  So we get the current scope,
    # and pretend it to be the root scope.
    scope = tf.get_variable_scope()
    old_name = scope.name
    try:
        scope._name = ''
        with variable_scope_ops._pure_variable_scope(
                '',
                reuse=reuse,
                initializer=initializer,
                regularizer=regularizer,
                caching_device=caching_device,
                partitioner=partitioner,
                custom_getter=custom_getter,
                old_name_scope='',
                dtype=dtype) as vs:
            scope._name = old_name
            with tf.name_scope(None):
                yield vs
    finally:
        scope._name = old_name


def lagacy_default_name_arg(method):
    """Add legacy support for `default_name` argument."""
    @six.wraps(method)
    def wrapper(*args, **kwargs):
        if 'default_name' in kwargs:
            warnings.warn('`default_name` is deprecated, please use `name` '
                          'instead.', category=DeprecationWarning)
            if 'scope' in kwargs:
                raise ValueError('`default_name` and `scope` cannot '
                                 'be specified at the same time.')
            if 'name' in kwargs:
                kwargs['scope'] = kwargs.pop('name')
            kwargs['name'] = kwargs.pop('default_name')
        return method(*args, **kwargs)
    return wrapper


class VarScopeObject(object):
    """Base class for object that owns a variable scope.

    Such an object is compatible with the use patterns of `instance_reuse`.

    Parameters
    ----------
    name : str
        Name of this object.

        When `scope` is not specified, the object will obtain a variable scope
        with unique name, according to the name suggested by `name`.
        If `scope` is also not specified, it will use the underscored
        class name as the default name.

        This argument will be stored and can be accessed via `name` attribute.
        (And if it will be None if not specified in construction argument)

    scope : str
        Scope of this object.

        Note that if `scope` is specified, the variable scope of constructed
        object will take exactly this scope, even if another object has already
        taken such scope.  That is to say, these two objects will share the same
        variable scope.
    """

    @lagacy_default_name_arg
    def __init__(self, name=None, scope=None):
        scope = scope or None
        name = name or None

        if not scope and not name:
            default_name = camel_to_underscore(self.__class__.__name__)
            default_name = default_name.lstrip('_')
        else:
            default_name = name

        with tf.variable_scope(scope, default_name=default_name) as vs:
            self._variable_scope = vs       # type: tf.VariableScope
            self._name = name

    @property
    def name(self):
        """Get the name of this object."""
        return self._name

    @property
    def variable_scope(self):
        """Get the variable scope of this object."""
        return self._variable_scope

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.variable_scope.name)


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
