# -*- coding: utf-8 -*-
import functools
import tensorflow as tf
from tensorflow.contrib import layers

from .base import Lambda

__all__ = [
    'Dense',
    'Linear',
]


class Dense(Lambda):
    """Dense layer component.

    This class wraps `layers.fully_connected` function into a dense layer,
    which reuses the weights and biases across multiple calls.

    Parameters
    ----------
    num_outputs : int
        The number of output units in the layer.

    activation_fn : any
        Activation function (default `tf.nn.relu`).
        Explicitly set it to None to skip it and maintain a linear activation.

    normalizer_fn : any
        Normalization function to apply.  If `normalizer_fn` is provided,
        then `biases_initializer` and `biases_regularizer` are ignored and
        `biases` are not created nor added. (default None)

    normalizer_params : dict[str, any]
        Normalization function named parameters. (default None)

    weights_initializer : any
        An initializer for the weights. (default `layers.xavier_initializer`)

    weights_regularizer : any
        Optional regularizer for the weights. (default None)

    biases_initializer : any
        An initializer for the biases. If None skip biases.
        (default `tf.zero_initializer`)

    biases_regularizer : any
        Optional regularizer for the biases. (default None)

    trainable : bool
        Whether or not to add the variables to the graph collection
        `tf.GraphKeys.TRAINABLE_VARIABLES`? (default True)

    name, default_name : str
        Name and default name of this linear component.
    """

    def __init__(self,
                 num_outputs,
                 activation_fn=tf.nn.relu,
                 normalizer_fn=None,
                 normalizer_params=None,
                 weights_initializer=layers.xavier_initializer(),
                 weights_regularizer=None,
                 biases_initializer=tf.zeros_initializer(),
                 biases_regularizer=None,
                 trainable=True,
                 name=None,
                 default_name=None):
        super(Dense, self).__init__(
            functools.partial(
                layers.fully_connected,
                num_outputs=num_outputs,
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params,
                weights_initializer=weights_initializer,
                weights_regularizer=weights_regularizer,
                biases_initializer=biases_initializer,
                biases_regularizer=biases_regularizer,
                trainable=trainable,
            ),
            name=name,
            default_name=default_name
        )


class Linear(Dense):
    """Linear layer component.

    This class wraps `layers.fully_connected` function into a linear layer,
    which reuses the weights and biases across multiple calls.

    Parameters
    ----------
    num_outputs : int
        The number of output units in the layer.

    weights_initializer
        An initializer for the weights. (default `layers.xavier_initializer`)

    weights_regularizer
        Optional regularizer for the weights.

    biases_initializer
        An initializer for the biases. If None skip biases.
        (default `tf.zero_initializer`)

    biases_regularizer
        Optional regularizer for the biases.

    trainable : bool
        Whether or not to add the variables to the graph collection
        `tf.GraphKeys.TRAINABLE_VARIABLES`? (default True)

    name, default_name : str
        Name and default name of this linear component.
    """

    def __init__(self,
                 num_outputs,
                 weights_initializer=layers.xavier_initializer(),
                 weights_regularizer=None,
                 biases_initializer=tf.zeros_initializer(),
                 biases_regularizer=None,
                 trainable=True,
                 name=None,
                 default_name=None):
        super(Linear, self).__init__(
            num_outputs=num_outputs,
            activation_fn=None,
            weights_initializer=weights_initializer,
            weights_regularizer=weights_regularizer,
            biases_initializer=biases_initializer,
            biases_regularizer=biases_regularizer,
            trainable=trainable,
            name=name,
            default_name=default_name,
        )
