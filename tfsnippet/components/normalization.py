# -*- coding: utf-8 -*-
import six
import tensorflow as tf
from tensorflow.python.layers.normalization import batch_normalization

from .base import Component

__all__ = [
    'BatchNorm'
]


class BatchNorm(Component):
    """Batch Normalization layer.

    A Batch Normalization layer should be applied between linear activation
    and its successive non-linear activation function, as is proposed in [1].

    [1]	S. Ioffe and C. Szegedy, “Batch Normalization - Accelerating Deep
        Network Training by Reducing Internal Covariate Shift,” arXiv,
        vol. cs.LG, 2015.

    Parameters
    ----------
    axis : int
        At which dimension should Batch Normalization be applied? (default -1)

    decay : float
        Decay for moving average. (default 0.99)

    epsilon : float
        Small float added to variance to avoid division by zero. (default 1e-3)

    center : bool
        Whether or not add `beta` to normalized tensor? (default True)

    scale : bool
        Whether or not multiply `gamma` to normalized tensor? (default False)

        This could be set to False if next layer is linear (also `tf.nn.relu`),
        since the scaling can be done by the next layer.

    beta_initializer
        Optional initializer for beta. (default ``tf.zeros_initializer()``)

    gamma_initializer
        Optional initializer for gamma. (default ``tf.ones_initializer()``)

    moving_mean_initializer
        Optional initializer for moving mean.
        (default ``tf.zeros_initializer()``)

    moving_variance_initializer
        Optional initializer for moving variance.
        (default ``tf.ones_initializer()``)

    beta_regularizer
        Optional regularizer for beta. (default None)

    gamma_regularizer
        Optional regularizer for gamma. (default None)

    is_training : bool | tf.Tensor
        Whether or not the layer is applied during training phase?
        (default True)

    trainable : bool
        If `True` also add variables to the graph collection
        `tf.GraphKeys.TRINABLE_VARIABLES`. (default True)

    name, default_name : str
        Name and default name of the component.
    """

    def __init__(self,
                 axis=-1,
                 decay=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer=tf.zeros_initializer(),
                 gamma_initializer=tf.ones_initializer(),
                 moving_mean_initializer=tf.zeros_initializer(),
                 moving_variance_initializer=tf.ones_initializer(),
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 is_training=True,
                 trainable=True,
                 name=None,
                 default_name=None):
        if not isinstance(axis, six.integer_types):
            raise TypeError('`axis` is expected to be an integer, '
                            'but got %r.' % (axis,))

        self.axis = axis
        self.decay = decay
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.is_training = is_training
        self.trainable = trainable
        super(BatchNorm, self).__init__(name=name, default_name=default_name)

    def _call(self, inputs, is_training=None):
        """Apply Batch Normalization on `inputs`.

        Parameters
        ----------
        inputs : tf.Tensor | tf.Variable
            The tensor upon which Batch Normalization should be applied.

        is_training : bool
            If True or False is specified, will overwrite the `is_training`
            argument specified in constructor. (default None)

        Returns
        -------
        tf.Tensor
            The Batch-Normalized tensor.
        """
        if is_training is None:
            is_training = self.is_training
        return batch_normalization(
            inputs,
            axis=self.axis,
            momentum=self.decay,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            beta_initializer=self.beta_initializer,
            gamma_initializer=self.gamma_initializer,
            moving_mean_initializer=self.moving_mean_initializer,
            moving_variance_initializer=self.moving_variance_initializer,
            beta_regularizer=self.beta_regularizer,
            gamma_regularizer=self.gamma_regularizer,
            trainable=self.trainable,
            training=is_training,
        )
