# -*- coding: utf-8 -*-
from contextlib import contextmanager

import six
import tensorflow as tf

from tfsnippet.utils import AutoReprObject, ContextStack, get_global_step

__all__ = [
    'OptionDefaults', 'option_defaults', 'get_option_defaults',
]


class OptionDefaults(AutoReprObject):
    """Scope to manage the default values for hyper-parameters.

    When training and evaluating a neural network model, there are usually
    tons of hyper-parameters to set.  This class provides a centralized way
    to provide default values for these hyper-parameters.

    Parameters
    ----------
    optimizer_factory : () -> tf.train.Optimizer
        Type of the training optimizer, or a factory to create the optimizer.
        
    global_step : tf.Tensor
        The default global step tensor to use.
        
    batch_size : int
        Default training batch size.

    predict_batch_size : int
        Default predicting batch size.

    gradient_clip_by_norm : float
        If specified, clip the gradient by this norm.

    gradient_clip_by_value : float
        If specified, clip the gradient by range [-value, value].

        When both of `gradient_clip_by_norm` and `gradient_clip_by_value`
        are specified, the gradient should be clipped first by norm, then
        by value at each dimension.
        
    weights_regularizer : (list[tf.Tensor]) -> tf.Tensor
        Function to compute the weights regularizer.
        
    biases_regularizer : (list[tf.Tensor]) -> tf.Tensor
        Function to compute the biases regularizer.
    """

    _initialized = False

    def __init__(self,
                 optimizer_factory=None,
                 global_step=None,
                 batch_size=None,
                 predict_batch_size=None,
                 gradient_clip_by_norm=None,
                 gradient_clip_by_value=None,
                 weights_regularizer=None,
                 biases_regularizer=None):
        self.optimizer_factory = optimizer_factory
        self.global_step = global_step
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size
        self.gradient_clip_by_norm = gradient_clip_by_norm
        self.gradient_clip_by_value = gradient_clip_by_value
        self.weights_regularizer = weights_regularizer
        self.biases_regularizer = biases_regularizer
        self._initialized = True

    def __setattr__(self, key, value):
        if self._initialized:
            raise AttributeError(
                'OptionDefaults object is read-only.  '
                'You should construct a new OptionDefaults object by '
                '`copy()` or `merge()`.'
            )
        return object.__setattr__(self, key, value)

    @contextmanager
    def as_default(self):
        """Open a scoped context with default parameters."""
        _defaults_stack.push(self)
        try:
            yield self
        finally:
            _defaults_stack.pop()

    def as_dict(self):
        """Get the default parameters as dict.
        
        Only the configured parameters (whose values are not None) will be
        included in the returned dict.
        
        Returns
        -------
        dict[str, any]
            The parameter value dict.
        """
        return {
            k: v for k, v in six.iteritems(self.__dict__)
            if not k.startswith('_') and v is not None
        }

    def copy(self, **kwargs):
        """Copy a new instance, with some parameters set to new values.

        Parameters
        ----------
        **kwargs
            The new parameters to set.

        Returns
        -------
        OptionDefaults
            The copied default parameters.
        """
        config_values = self.as_dict()
        config_values.update(kwargs)
        return OptionDefaults(**config_values)

    def merge(self, other):
        """Merge the parameters with another `OptionDefaults` instance.
        
        The parameters will take values first from `other`, and then those
        not configured in `other` will take values from this instance.
        
        Parameters
        ----------
        other : OptionDefaults
            The other default parameters object.
        
        Returns
        -------
        OptionDefaults
            The merged default parameters.
        """
        config_values = self.as_dict()
        config_values.update(other.as_dict())
        return OptionDefaults(**config_values)

    def get_optimizer(self):
        """Get an optimizer according to default parameters.
        
        If the `optimizer_factory` has not been set, return an instance
        of `tf.train.AdamOptimizer` with default arguments.
        
        Returns
        -------
        tf.train.Optimizer
            The constructed optimizer object.
        """
        if self.optimizer_factory is None:
            return tf.train.AdamOptimizer()
        return self.optimizer_factory()

    def get_global_step(self):
        """Get the global step tensor.
        
        If the global step tensor is not configured, will get one
        using `tfsnippet.utils.get_global_step`.
        
        Returns
        -------
        tf.Tensor
            The global step tensor.
        """
        if self.global_step is not None:
            return self.global_step
        return get_global_step()

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


@contextmanager
def option_defaults(**kwargs):
    """Open a scoped context with default parameters.
    
    The specified default parameters will be merged with the default
    parameters in current context.
    
    Parameters
    ----------
    **kwargs
        Default parameters to set.
    """
    with get_option_defaults().copy(**kwargs).as_default() as defaults:
        yield defaults


def get_option_defaults():
    """Get the default parameters object.
    
    Returns
    -------
    OptionDefaults
        The default parameters object.
    """
    return _defaults_stack.top()

_defaults_stack = ContextStack(initial_factory=OptionDefaults)
