# -*- coding: utf-8 -*-
from tfsnippet.utils import VarScopeObject, instance_reuse

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

    @instance_reuse(scope='__call__')
    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)
