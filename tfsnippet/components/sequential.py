# -*- coding: utf-8 -*-
import tensorflow as tf

from .base import Component

__all__ = ['Sequential']


class Sequential(Component):
    """Sequential neural network wrapper.

    It is a common task to build a sequential of neural network layers,
    sharing the same set of parameters.  With the help of `Component`,
    one may reuse variables introduced by layer builders from a wide
    variety of third-party libraries, simply by calling these builders
    within `_call` method.

    This class provides a common wrapper upon these layer builders of
    sequential networks.  Following code demonstrates an example of
    using this wrapper:

        from tfsnippet.components import Linear

        mlp = Sequential([
            Linear(100),
            tf.nn.relu,
            Linear(100),
            tf.nn.relu,
            Linear(1),
            tf.nn.sigmoid
        ])

    Which builds a multi-layer perceptron, with 2 hidden layers of 100
    units and ReLU activation, plus a sigmoid output layer with 1 unit,
    sharing parameter variables among multiple instances of this MLP.

    Note that if another instance of `Component` is specified as one
    layer, the variables of that component is managed within its scope,
    instead of the scope of this sequential component.

    If instead a function or a method is provided, it will be called
    within the child variable scope of "layer%d", under the scope of
    sequential component itself.

    Parameters
    ----------
    layers : list[(inputs) -> outputs]
        Layers of this sequential component, each should consume the
        outputs of previous layer as inputs (the first layer should
        accept the inputs given to the whole sequential component).

        The outputs of the last layer will be the outputs of the
        whole sequential component.

    name, default_name : str
        Name and default name of this sequential component.
    """

    def __init__(self, layers, name=None, default_name=None, temp_outputs=None):
        layers = tuple(layers)
        if not layers:
            raise ValueError('`components` must not be empty.')
        super(Sequential, self).__init__(name=name, default_name=default_name)
        self._layers = layers
        self._temp_outputs = temp_outputs

    def _call(self, inputs):
        if self._temp_outputs is not None:
            if not isinstance(self._temp_outputs, list) or \
                            len(self._temp_outputs) > 0:
                raise ValueError('`temp_outputs` must be a empty list.')
            outputs = self._temp_outputs
        else:
            outputs = []
        outputs.append(inputs)
        for i, c in enumerate(self._layers):
            with tf.variable_scope('layer%d' % i):
                outputs.append(c(outputs[i]))
        return outputs[-1]
