# -*- coding: utf-8 -*-
import inspect
import functools
import weakref
from contextlib import contextmanager

import six
import tensorflow as tf

from .scope import reopen_variable_scope, root_variable_scope

__all__ = [
    'auto_reuse_variables', 'local_reuse', 'global_reuse', 'instance_reuse',
]


@contextmanager
def auto_reuse_variables(name_or_scope,
                         initializer=None,
                         regularizer=None,
                         caching_device=None,
                         partitioner=None,
                         custom_getter=None,
                         dtype=tf.float32):
    """Open a variable scope, while automatically choosing `reuse` flag.

    The `reuse` flag will be set to False if the variable scope is opened
    for the first time, and it will be set to True each time the variable
    scope is opened again.

    Parameters
    ----------
    name_or_scope : str | tf.VariableScope
        The name of the variable scope, or the variable scope to open.

    initializer, regularizer, caching_device, partitioner, custom_getter, dtype
        Other parameters for opening the variable scope.

    Yields
    ------
    tf.VariableScope
        The opened variable scope.
    """
    if not name_or_scope:
        raise ValueError('`name_or_scope` cannot be empty.  If you want to '
                         'auto-reuse variables in root variable scope, you '
                         'should capture the root variable scope instance '
                         'and call `auto_reuse_variables` on that, instead '
                         'of calling with an empty name.')

    with tf.variable_scope(name_or_scope,
                           initializer=initializer,
                           regularizer=regularizer,
                           caching_device=caching_device,
                           partitioner=partitioner,
                           custom_getter=custom_getter,
                           dtype=dtype) as vs:
        # check whether or not the variable scope has been initialized
        graph = tf.get_default_graph()
        if graph not in __auto_reuse_variables_graph_dict:
            __auto_reuse_variables_graph_dict[graph] = set([])
        initialized_scopes = __auto_reuse_variables_graph_dict[graph]
        reuse = vs.name in initialized_scopes

        # if `reuse` is True, set the reuse flag
        if reuse:
            vs.reuse_variables()
            yield vs
        else:
            yield vs
            initialized_scopes.add(vs.name)

#: dict to track the initialization state for each variable scope
#: belonging to every living graph.
__auto_reuse_variables_graph_dict = weakref.WeakKeyDictionary()


def local_reuse(method=None, scope=None):
    """Decorate a function within `auto_reuse_variables` scope locally.

    Any function or method applied with this decorator will be called within
    a variable scope opened by `auto_reuse_variables`. That is, the following
    code:

        @local_reuse
        def foo():
            return tf.get_variable('bar', ...)

        bar = foo()

    is equivalent to:

        with auto_reuse_variables('foo'):
            bar = tf.get_variable('bar', ...)

    Note that the scope opened by `auto_reuse_variables` should be child
    of the current opened variable scope, so that the following variables,
    `bar_1` and `bar_2`, should be different variables, since they are
    created within different variable scopes:

        with tf.variable_scope('parent_1'):
            bar_1 = foo()       # bar_1.name == 'parent_1/foo/bar:0'

        with tf.variable_scope('parent_2'):
            bar_2 = foo()       # bar_2.name == 'parent_2/foo/bar:0'

    By default the name of the variable scope should be equal to the name
    of the decorated method, and the name scope within the context should
    be equal to the variable scope name, plus some suffix to make it unique.
    The variable scope name can be set by `scope` argument, for example:

        @local_reuse(scope='dense')
        def dense_layer(inputs):
            w = tf.get_variable('w', ...)
            b = tf.get_variable('b', ...)
            return tf.matmul(w, inputs) + b

    Note that the variable reusing is based on the name of the variable
    scope, rather than the function object.  As a result, two functions
    with the same name, or with the same `scope` argument, will reuse
    the same set of variables.  For example:

        @local_reuse(scope='foo')
        def foo_1():
            return tf.get_variable('bar', ...)

        @local_reuse(scope='foo')
        def foo_2():
            return tf.get_variable('bar', ...)

    These two functions will return the same `bar` variable.

    See Also
    --------
    global_reuse, instance_reuse, auto_reuse_variables

    Parameters
    ----------
    scope : str
        The name of the variable scope.  If not set, will use the name
        of the method as scope name.
    """
    if method is None:
        return functools.partial(local_reuse, scope=scope)
    scope = scope or method.__name__

    @six.wraps(method)
    def wrapper(*args, **kwargs):
        with auto_reuse_variables(scope):
            return method(*args, **kwargs)

    return wrapper


def global_reuse(method=None, scope=None):
    """Decorate a function within `auto_reuse_variables` scope globally.

    Any function or method applied with this decorator will be called within
    a variable scope opened first by `root_variable_scope`, then by
    `auto_reuse_variables`. That is, the following code:

        @global_reuse
        def foo():
            return tf.get_variable('bar', ...)

        bar = foo()

    is equivalent to:

        with root_variable_scope():
            with auto_reuse_variables('foo'):
                bar = tf.get_variable('bar', ...)

    Thus the major difference between `global_reuse` and `local_reuse` is
    that `global_reuse` will not follow the caller's active variable scope.

    By default the name of the variable scope should be equal to the name
    of the decorated method, and the name scope within the context should
    be equal to the variable scope name, plus some suffix to make it unique.
    The variable scope name can be set by `scope` argument, for example:

        @global_reuse(scope='dense')
        def dense_layer(inputs):
            w = tf.get_variable('w', ...)
            b = tf.get_variable('b', ...)
            return tf.matmul(w, inputs) + b

    Note that the variable reusing is based on the name of the variable
    scope, rather than the function object.  As a result, two functions
    with the same name, or with the same `scope` argument, will reuse
    the same set of variables.  For example:

        @global_reuse(scope='foo')
        def foo_1():
            return tf.get_variable('bar', ...)

        @global_reuse(scope='foo')
        def foo_2():
            return tf.get_variable('bar', ...)

    These two functions will return the same `bar` variable.

    See Also
    --------
    local_reuse, instance_reuse,auto_reuse_variables

    Parameters
    ----------
    scope : str
        The name of the variable scope.  If not set, will use the name
        of the method as scope name.
    """
    if method is None:
        return functools.partial(local_reuse, scope=scope)
    scope = scope or method.__name__

    @six.wraps(method)
    def wrapper(*args, **kwargs):
        with root_variable_scope():
            with auto_reuse_variables(scope):
                return method(*args, **kwargs)

    return wrapper


def instance_reuse(method=None, scope=None):
    """Decorate an instance method within `auto_reuse_variables` scope.

    This decorator should be applied to unbound instance methods, and
    the instances that owns the methods are expected to have `variable_scope`
    attribute.  For example:

        class Foo(object):

            def __init__(self, name):
                with tf.variable_scope(name) as vs:
                    self.variable_scope = vs

            @instance_reuse
            def foo(self):
                return tf.get_variable('bar', ...)

    The above example is then equivalent to the following code:

        class Foo(object):

            def __init__(self, name):
                with tf.variable_scope(name) as vs:
                    self.variable_scope = vs

            def foo(self):
                with reopen_variable_scope(self.variable_scope):
                    with auto_reuse_variables('foo'):
                        return tf.get_variable('bar', ...)

    In which the `instance_reuse` decorator acts like `global_reuse`,
    but will open the `variable_scope` of corresponding instance instead
    of opening the root variable scope, before entering the desired
    auto-reusing variable scope.

    See Also
    --------
    global_reuse, local_reuse, auto_reuse_variables

    Parameters
    ----------
    scope : str
        The name of the variable scope.  If not set, will use the name
        of the method as scope name.
    """

    if method is None:
        return functools.partial(instance_reuse, scope=scope)

    # check whether or not `method` looks like an instance method
    if six.PY2:
        getargspec = inspect.getargspec
    else:
        getargspec = inspect.getfullargspec

    argspec = getargspec(method)
    if argspec.args[0] != 'self':
        raise TypeError('`method` seems not to be an instance method '
                        '(whose first argument should be `self`).')
    if inspect.ismethod(method):
        raise TypeError('`method` is expected to be unbound instance method.')

    # determine the scope name
    scope = scope or method.__name__

    @six.wraps(method)
    def wrapper(*args, **kwargs):
        obj = args[0]
        variable_scope = obj.variable_scope
        if not isinstance(variable_scope, tf.VariableScope):
            raise TypeError('`variable_scope` attribute of the instance %r '
                            'is expected to be a `tf.VariableScope`, but got '
                            '%r.' % (obj, variable_scope,))

        with reopen_variable_scope(variable_scope):
            with auto_reuse_variables(scope):
                return method(*args, **kwargs)

    return wrapper
