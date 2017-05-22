# -*- coding: utf-8 -*-
from tfsnippet import distributions
from .base import StochasticTensor

__all__ = [
    'Categorical',
    'OneHotCategorical',
    'Discrete',
    'OneHotDiscrete',
]


class Categorical(StochasticTensor):
    """Stochastic tensor for Categorical distribution.

    Parameters
    ----------
    logits : tf.Tensor | np.ndarray
        A float tensor of shape (..., n_categories), which is the
        un-normalized log-odds of probabilities of the categories.

    probs : tf.Tensor | np.ndarray
        A float tensor of shape (..., n_categories), which is the
        normalized probabilities of the categories.

        One and only one of `logits` and `probs` should be specified.
        The relationship between these two arguments, if given each other,
        is stated as follows:

            .. math::
                \\begin{aligned}
                    \\text{logits} &= \\log (\\text{probs}) \\\\
                     \\text{probs} &= \\text{softmax} (\\text{logits})
                \\end{aligned}

    dtype : tf.DType | np.dtype | str
        The data type of samples from the distribution. (default is `tf.int32`)

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

    def __init__(self, logits=None, probs=None, dtype=None, sample_num=None,
                 observed=None, validate_observed_shape=False,
                 group_event_ndims=None, name=None, default_name=None):
        super(Categorical, self).__init__(
            distribution=lambda: distributions.Categorical(
                logits=logits,
                probs=probs,
                dtype=dtype,
                name='distribution'
            ),
            sample_num=sample_num,
            observed=observed,
            validate_observed_shape=validate_observed_shape,
            group_event_ndims=group_event_ndims,
            name=name,
            default_name=default_name,
        )

Discrete = Categorical


class OneHotCategorical(StochasticTensor):
    """Stochastic tensor for one-hot encoded Categorical distribution.

    Parameters
    ----------
    logits : tf.Tensor | np.ndarray
        A float tensor of shape (..., n_categories), which is the
        un-normalized log-odds of probabilities of the categories.

    probs : tf.Tensor | np.ndarray
        A float tensor of shape (..., n_categories), which is the
        normalized probabilities of the categories.

        One and only one of `logits` and `probs` should be specified.
        The relationship between these two arguments, if given each other,
        is stated as follows:

            .. math::
                \\begin{aligned}
                    \\text{logits} &= \\log (\\text{probs}) \\\\
                     \\text{probs} &= \\text{softmax} (\\text{logits})
                \\end{aligned}

    dtype : tf.DType | np.dtype | str
        The data type of samples from the distribution. (default is `tf.int32`)

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

    def __init__(self, logits=None, probs=None, dtype=None, sample_num=None,
                 observed=None, validate_observed_shape=False,
                 group_event_ndims=None, name=None, default_name=None):
        super(OneHotCategorical, self).__init__(
            distribution=lambda: distributions.OneHotCategorical(
                logits=logits,
                probs=probs,
                dtype=dtype,
                name='distribution'
            ),
            sample_num=sample_num,
            observed=observed,
            validate_observed_shape=validate_observed_shape,
            group_event_ndims=group_event_ndims,
            name=name,
            default_name=default_name,
        )

OneHotDiscrete = OneHotCategorical
