# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.components import Component

__all__ = ['StochasticLayer']


class StochasticLayer(Component):
    """Base class for all stochastic layers.

    For every distributions in the package `tfsnippet.bayes.distributions`,
    we provide one or more `StochasticLayer`, which derives the distribution
    parameters from the tensor(s) feeding into the `StochasticLayer`, in the
    most proper way, and produces a `StochasticTensor` as the corresponding
    output of the layer.

    These specialized `StochasticLayer` classes could be used as components
    when building realistic generative models, along with the common network
    components like `tfsnippet.components.Sequential`.

    Parameters
    ----------
    n_samples : int | tf.Tensor | None
        Generate this number of samples via `sample_n` method, unless
        `observed` is specified.  Default is None, indicating just one
        sample without the auxiliary sample dimension.

        See `Distribution.sample_n` and `Distribution.sample_or_observe`
        for more details.

    observed : tf.Tensor | np.ndarray | float | int
        The observations.  If specified, will use the observations as
        output, instead sampling from the distribution. (default None)

    group_event_ndims : int | tf.Tensor
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

    name, default_name : str
        Name and default name of this stochastic layer.

    See Also
    --------
    Distribution.sample_n, Distribution.sample_or_observe
    """

    def __init__(self, n_samples=None, observed=None, group_event_ndims=None,
                 name=None, default_name=None):
        super(StochasticLayer, self).__init__(name=name,
                                              default_name=default_name)
        self._n_samples = n_samples
        self._observed = observed
        self._group_event_ndims = group_event_ndims

    @property
    def n_samples(self):
        """Get the number of samples to take from the distribution.

        Returns
        -------
        int | tf.Tensor | None
            The number of samples.  If `n_samples` is not specified
            in the constructor, will return None.
        """
        return self._n_samples

    @property
    def observed(self):
        """Get the observations of this `StochasticLayer`.

        Returns
        -------
        Any
            The observations.
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

    def __call__(self, *args, **kwargs):
        """Build the output of this `StochaticLayer`.

        Parameters
        ----------
        *args
            Arguments for building the output.

        n_samples : int | tf.Tensor | None
            If specified, override the `n_samples` in the constructor.

        observed : tf.Tensor | np.ndarray | float | int
            If specified, override the `observed` in the constructor.

        group_event_ndims : int | tf.Tensor
            If specified, override the `group_event_ndims` in the
            constructor.

        **kwargs
            Other named arguments for building the output.

        Returns
        -------
        StochasticTensor
            The stochastic tensor produced by this layer.
        """
        kwargs.setdefault('n_samples', self.n_samples)
        kwargs.setdefault('observed', self.observed)
        kwargs.setdefault('group_event_ndims', self.group_event_ndims)
        return super(StochasticLayer, self).__call__(*args, **kwargs)
