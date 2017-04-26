# -*- coding: utf-8 -*-
from tfsnippet.utils import ScopedObject, open_variable_scope

__all__ = ['BaseComponent']


class BaseComponent(ScopedObject):
    """Base class for neural network components.
    
    A neural network component is basically a reusable object that
    could derive outputs for inputs, using the same, fixed set of
    parameters all the time.
    
    In order to implement such a component, derived classes should 
    implement a `_build` method, which takes the inputs and compute outputs.
    The `_build` method will be called by `BaseComponent.__call__` each
    time the component object is being used, and a proper variable scope
    (with `reuse` flag automatically set) will be opened before calling
    `_build` method.
    """

    def _build(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        with open_variable_scope(self.variable_scope):
            return self._build(*args, **kwargs)
