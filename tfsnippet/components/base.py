# -*- coding: utf-8 -*-
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
        
            def _build(self, inputs):
                return layers.fully_connected(inputs, num_outputs=2)
    """

    def _build(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        with auto_reuse_variables(self.variable_scope):
            return self._build(*args, **kwargs)
