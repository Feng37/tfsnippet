# -*- coding: utf-8 -*-
from tfsnippet import distributions
from .base import StochasticTensor

__all__ = ['Gamma']


class Gamma(StochasticTensor):
    """Stochastic tensor for Gamma distribution.

    Parameters
    ----------
    alpha : tf.Tensor | np.ndarray | float
        The shape parameter of the Gamma distribution.
        Should be positive and broadcastable to match `beta`.

        Note that the range of `alpha` is :math:`(0, \\infty)`.

    beta : tf.Tensor | np.ndarray | float
        The inverse scale parameter of the Gamma distribution.
        Should be positive and broadcastable to match `alpha`.
        Also, the data type of `beta` will be casted to that of `alpha`.

        Note that the range of `beta` is :math:`(0, \\infty)`.

    sample_num : int | tf.Tensor | tf.Variable
        Optional number of samples to take. (default None)

    observed : tf.Tensor
        If specified, random sampling will not take place.
        Instead this tensor will be regarded as sampled tensor.
        (default None)

    validate_observed_shape : bool
        Whether or not to validate the shape of `observed`? (default False)

        If set to True, the shape of `observed` must match that returned
        by `get_shape()`, given `sample_num` argument.

    group_event_ndims : int
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

    name : str
        Name of this normal distribution.

    default_name : str
        Default name of this normal distribution.
    """

    def __init__(self, alpha, beta, sample_num=None,
                 observed=None, validate_observed_shape=False,
                 group_event_ndims=None, name=None, default_name=None):
        super(Gamma, self).__init__(
            distribution=lambda: distributions.Gamma(
                alpha=alpha,
                beta=beta,
                name='distribution'
            ),
            sample_num=sample_num,
            observed=observed,
            validate_observed_shape=validate_observed_shape,
            group_event_ndims=group_event_ndims,
            name=name,
            default_name=default_name,
        )
