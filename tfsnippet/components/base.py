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
    
        from tensorflow.contrib import layers

        class MyComponent(BaseComponent):
        
            def _build(self, inputs):
                return layers.fully_connected(
                    inputs, 
                    num_outputs=2,
                    scope='fully_connected'
                )

    Note that the `scope` argument in the above example is necessary,
    in which it enables the fully connected layer to use the same set
    of parameters every time when `_build` is called.
    """

    def _build(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        with auto_reuse_variables(self.variable_scope, unique_name_scope=True):
            return self._build(*args, **kwargs)
