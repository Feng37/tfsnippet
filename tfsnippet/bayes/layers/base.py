# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from tfsnippet.components import Component
from ..stochastic import StochasticTensor

__all__ = ['StochasticLayer']


class StochasticLayer(Component):
    """Base class for stochastic layers.

    A `StochasticLayer` should act as a thin wrapper for `Distribution`,
    which can be used as a network `Component` along with other components
    like `Sequential` and `Linear`.

    Basically, it derives a `StochasticTensor` according to layer inputs.
    The layer inputs should typically be a dict of distribution parameters,
    while the sampling parameters might be specified in the constructor,
    for example:

        x_layer = NormalLayer(n_samples=100)
        x = x_layer({'mean': x_mean, 'logstd': x_logstd})

    These sampling parameters may also be overwritten in the layer inputs,
    for example:

        x_layer = NormalLayer()
        x = x_layer({'mean': x_mean, 'logstd': x_logstd, 'n_samples': 100})

    Besides, the distribution and sampling parameters may also be fed into
    a `StochasticLayer` by named arguments, for example:

        x_layer = NormalLayer()
        x = x_layer(mean=x_mean, logstd=x_logstd, n_samples=100)

    The three different types of parameters (the layer inputs, the named
    arguments and the constructor arguments) have the following order of
    priorities:

        layer inputs > named arguments > constructor arguments

    That is to say, in the following example, where `n_samples` is specified
    for three times, `x` will finally take 30 random samples:

        x_layer = NormalLayer(n_samples=10)
        x = x_layer({..., 'n_samples': 30}, n_samples=20)

    Parameters
    ----------
    n_samples : int | tf.Tensor | None
        The number of samples to take. (default None)

        See `Distribution.sample_n` and `Distribution.sample_or_observe`
        for more details.

    observed : tf.Tensor | np.ndarray
        The observation of this stochastic layer. (default None)

    group_event_ndims : int | tf.Tensor
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

        See `Distribution`, `StochasticTensor` for more details.

    name, default_name : str
        Optional name or default name of this `StochasticLayer`.
    """

    def __init__(self, n_samples=None, observed=None, group_event_ndims=None,
                 name=None, default_name=None):
        self._n_samples = n_samples
        self._observed = observed
        self._group_event_ndims = group_event_ndims
        super(StochasticLayer, self).__init__(name=name,
                                              default_name=default_name)

    @property
    def n_samples(self):
        """Get the number of samples to take.

        Returns
        -------
        int | tf.Tensor | None
            The number of samples to take, or None if not specified.
        """
        return self._n_samples

    @property
    def observed(self):
        """Get the observed tensor.

        Returns
        -------
        tf.Tensor | np.ndarray | None
            The observed tensor, or None if not specified.
        """
        return self._observed

    @property
    def group_event_ndims(self):
        """Get the number of dimensions to be considered as events group.

        Returns
        -------
        int | tf.Tensor | None
            The number of dimensions.  If `group_event_ndims` is not
            specified in the constructor, will return None.
        """
        return self._group_event_ndims

    def __call__(self, inputs, **kwargs):
        """Derive the `StochasticTensor` output from this `StochasticLayer`.

        Parameters
        ----------
        inputs : dict[str, any]
            The distribution and sampling parameters.

        **kwargs
            Other named arguments as sampling parameters.

        Returns
        -------
        StochasticTensor
            The derived stochastic tensor.
        """
        if not isinstance(inputs, (dict, OrderedDict)):
            raise TypeError('`inputs` is expected to be a dict, but got %r.'
                            % (inputs,))
        kwargs.update(inputs)
        for k in ('n_samples', 'observed', 'group_event_ndims'):
            kwargs.setdefault(k, getattr(self, k))
        return self._call(**kwargs)
