# -*- coding: utf-8 -*-
import inspect
import functools
import weakref
from contextlib import contextmanager

import six
import tensorflow as tf

from .scope import open_variable_scope

__all__ = [
    'auto_reuse_variables', 'static_reuse', 'instance_reuse',
]


@contextmanager
def auto_reuse_variables(name_or_scope,
                         unique_name_scope=True,
                         **kwargs):
    """Open a variable scope, while automatically choosing `reuse` flag.

    The `reuse` flag will be set to False if the variable scope is opened
    for the first time, and it will be set to True each time the variable
    scope is opened again.

    Parameters
    ----------
    name_or_scope : str | tf.VariableScope
        The name of the variable scope, or the variable scope to open.

    unique_name_scope : bool
        Whether or not to open a unique name scope?

        If True, a unique name scope will be opened.
        Otherwise the original name scope of the variable scope will be
        reopened. (default is True)

    **kwargs
        Other parameters for opening the variable scope.

    Yields
    ------
    tf.VariableScope
        The opened variable scope.
    """
    if 'reuse' in kwargs:
        raise RuntimeError(
            '`reuse` is not an argument of `auto_reuse_variables`.')

    # open the variable scope temporarily to capture the actual scope
    with open_variable_scope(name_or_scope,
                             unique_name_scope=False,
                             pure_variable_scope=True) as vs:
        variable_scope = vs

    # check whether or not the variable scope has been initialized
    graph = tf.get_default_graph()
    if graph not in __auto_reuse_variables_graph_dict:
        __auto_reuse_variables_graph_dict[graph] = set([])
    initialized_scopes = __auto_reuse_variables_graph_dict[graph]
    reuse = variable_scope.name in initialized_scopes

    # re-enter the variable scope with proper `reuse` flag
    with open_variable_scope(variable_scope,
                             unique_name_scope=unique_name_scope,
                             reuse=reuse,
                             **kwargs) as vs:
        yield vs
        initialized_scopes.add(vs.name)

#: dict to track the initialization state for each variable scope
#: belonging to every living graph.
__auto_reuse_variables_graph_dict = weakref.WeakKeyDictionary()


def static_reuse(method=None, scope=None, unique_name_scope=True):
    """Decorate a function or a method within `auto_reuse_variables` scope.

    Any function or method applied with this decorator will be called within
    a variable scope opened by `auto_reuse_variables`. That is, the following
    code:

        @static_reuse
        def foo():
            return tf.get_variable('bar', ...)

        bar = foo()

    is equivalent to:

        with auto_reuse_variables('foo', unique_name_scope=True):
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
    These behaviors can be changed by setting `scope` and `unique_name_scope`
    arguments, for example:

        @static_reuse(scope='dense')
        def dense_layer(inputs):
            w = tf.get_variable('w', ...)
            b = tf.get_variable('b', ...)
            return tf.matmul(w, inputs) + b

    Note that the variable reusing is based on the name of the variable
    scope, rather than the function object.  As a result, two functions
    with the same name, or with the same `scope` argument, will reuse
    the same set of variables.  For example:

        @static_reuse(scope='foo')
        def foo_1():
            return tf.get_variable('bar', ...)

        @static_reuse(scope='foo')
        def foo_2():
            return tf.get_variable('bar', ...)

    These two functions will return the same `bar` variable.

    Parameters
    ----------
    scope : str
        The name of the variable scope.  If not set, will use the name
        of the method as scope name.

    unique_name_scope : bool
        Whether or not to open a unique name scope each time the method
        is called?  (default is True)
    """
    if method is None:
        return functools.partial(
            static_reuse, scope=scope, unique_name_scope=unique_name_scope)
    scope = scope or method.__name__

    @six.wraps(method)
    def wrapper(*args, **kwargs):
        with auto_reuse_variables(scope, unique_name_scope=unique_name_scope):
            return method(*args, **kwargs)

    return wrapper


def instance_reuse(method=None, scope=None, unique_name_scope=True):
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

        @static_reuse
        def foo():
            return tf.get_variable('bar', ...)


        class Foo(object):

            def __init__(self, name):
                with tf.variable_scope(name) as vs:
                    self.variable_scope = vs

            def foo(self):
                with open_variable_scope(self.variable_scope,
                                         unique_name_scope=False):
                    return foo()

    In which the `instance_reuse` decorator acts like `static_reuse`,
    but will open the `variable_scope` of corresponding instance before
    entering a reused variable scope.

    See `static_reuse` and `open_variable_scope` for more details.

    Parameters
    ----------
    scope : str
        The name of the variable scope.  If not set, will use the name
        of the method as scope name.

    unique_name_scope : bool
        Whether or not to open a unique name scope each time the method
        is called?  (default is True)
    """

    if method is None:
        return functools.partial(
            instance_reuse, scope=scope, unique_name_scope=unique_name_scope)

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

        with open_variable_scope(variable_scope, unique_name_scope=False):
            with auto_reuse_variables(
                    scope, unique_name_scope=unique_name_scope):
                return method(*args, **kwargs)

    return wrapper
