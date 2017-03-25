# -*- coding: utf-8 -*-
import six
import tensorflow as tf

__all__ = ['ParamDefaults']


class ParamDefaults(object):
    """Scope to manage the default values for hyper-parameters.

    When training and evaluating a neural network model, there are usually
    tons of hyper-parameters to set.  This class provides a centralized way
    to provide default values for these hyper-parameters.

    Parameters
    ----------
    optimizer_factory : () -> tf.train.Optimizer
        Type of the training optimizer, or a factory to create the optimizer.
        If not specified, will use an instance of `tf.train.AdamOptimizer`.

    batch_size : int
        Default training batch size.
        If not specified, will use `32` as the mini-batch size.

    predict_batch_size : int
        Default predicting batch size.
        If not specified, will predict in one pass, instead of in mini-batches.

    gradient_clip_by_norm : float
        If specified, clip the gradient by this norm.

    gradient_clip_by_value : float
        If specified, clip the gradient by range [-value, value].

        When both of `gradient_clip_by_norm` and `gradient_clip_by_value`
        are specified, the gradient should be clipped first by norm, then
        by value at each dimension.
    """

    def __init__(self,
                 optimizer_factory=None,
                 batch_size=32,
                 predict_batch_size=None,
                 gradient_clip_by_norm=None,
                 gradient_clip_by_value=None):
        self.optimizer_factory = optimizer_factory
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size
        self.gradient_clip_by_norm = gradient_clip_by_norm
        self.gradient_clip_by_value = gradient_clip_by_value

    def copy(self, **kwargs):
        """Copy a new instance, with some parameters set to new values.

        Parameters
        ----------
        **kwargs
            The new parameter values to set.

        Returns
        -------
        ParamDefaults
            The copied parameter default values.
        """
        config_values = {
            k: v for k, v in six.iteritems(self.__dict__)
            if not k.startswith('_')
        }
        config_values.update(**kwargs)
        return ParamDefaults(**config_values)

    def get_optimizer(self):
        """Get an optimizer according to default parameters.."""
        if self.optimizer_factory is None:
            return tf.train.AdamOptimizer()
        return self.optimizer_factory()

    def clip_gradient(self, grad_vars):
        """Clip the gradients by default parameters.

        Parameters
        ----------
        grad_vars : collections.Iterable[(tf.Tensor, tf.Variable)]
            The gradient-variable list, which should be produced by
            `optimizer.compute_gradients`.

        Returns
        -------
        list[(tf.Tensor, tf.Variable)]
            The clipped gradients along with the variables.
        """
        if self.gradient_clip_by_norm is not None:
            clip_norm = self.gradient_clip_by_norm
            grad_vars = [
                (tf.clip_by_norm(grad, clip_norm), var)
                for grad, var in grad_vars
            ]
        if self.gradient_clip_by_value is not None:
            clip_value = self.gradient_clip_by_norm
            clip_low, clip_high = -clip_value, clip_value
            grad_vars = [
                (tf.clip_by_value(grad, clip_low, clip_high), var)
                for grad, var in grad_vars
            ]
        return list(grad_vars)
