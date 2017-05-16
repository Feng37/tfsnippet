# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.utils import VarScopeObject, auto_reuse_variables

__all__ = [
    'Component', 'Lambda',
]


class Component(VarScopeObject):
    """Base class for neural network components.

    A neural network component is basically a reusable object that
    could derive outputs for inputs, using the same, fixed set of
    parameters every time.

    For example, one may implement a reusable neural network component
    as follows:

        class MyComponent(Component):

            def _call(self, inputs):
                return layers.fully_connected(inputs, num_outputs=2)
    """

    def _call(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        with auto_reuse_variables(self.variable_scope,
                                  reopen_name_scope=True):
            # Here `reopen_name_scope` is set to True, so that multiple
            # calls to the same Component instance will not occupy multiple
            # unique name scopes derived from the original name scope.
            #
            # However, in order for ``tf.variable_scope(default_name=...)``
            # to work properly along with variable reusing, we must generate
            # a nested unique name scope, using ``tf.name_scope('build')``
            with tf.name_scope('build'):
                return self._call(*args, **kwargs)


class Lambda(Component):
    """Neural network component wrapper for arbitrary function.

    This class can wrap an arbitrary function or lambda expression into
    a neural network component, which reuses the variables created within
    the context of the function.

    For example, one may create a `Lambda` component by:

        from tensorflow.contrib import layers

        comp = Lambda(lambda inputs: layers.fully_connected(
            inputs, num_outputs=100, activation_fn=tf.nn.relu
        ))

    Parameters
    ----------
    f : (*args, **kwargs) -> outputs
        The function or lambda expression that derives the outputs.

    name, default_name : str
        Name and default name of the lambda component.
    """

    def __init__(self, f, name=None, default_name=None):
        super(Lambda, self).__init__(name=name, default_name=default_name)
        self._factory = f

    def _call(self, *args, **kwargs):
        return self._factory(*args, **kwargs)
