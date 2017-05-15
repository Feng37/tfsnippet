# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.utils import VarScopeObject, auto_reuse_variables

__all__ = ['Component']


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
            # suffixes of the variable scope name.
            #
            # However, in order for ``tf.variable_scope(default_name=...)``
            # to work properly along with variable reusing, we must generate
            # a unique name scope, using ``tf.name_scope('build')``
            with tf.name_scope('build'):
                return self._call(*args, **kwargs)
