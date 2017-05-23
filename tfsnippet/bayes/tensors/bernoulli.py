# -*- coding: utf-8 -*-
from tfsnippet import distributions
from .base import StochasticTensor

__all__ = ['Bernoulli']


class Bernoulli(StochasticTensor):
    """Stochastic tensor for Bernoulli distribution.

    Parameters
    ----------
    logits : tf.Tensor | np.ndarray | float
        A float tensor, which is the log-odds of probabilities of being 1.
        Note that the range of `logits` is :math:`(-\\infty, \\infty)`.

    probs : tf.Tensor | np.ndarray | float
        A float tensor, which is the probabilities of being 1.

        One and only one of `logits` and `p` should be specified.
        The relationship between Bernoulli `p` and `logits` are:

            .. math::
                \\begin{aligned}
                    \\text{logits} &= \\log \\frac{p}{1-p} \\\\
                                 p &= \\frac{1}{1 + \\exp(-\\text{logits})}
                \\end{aligned}

        Note that the range of `probs` is :math:`[0, 1]`.

    dtype : tf.DType | np.dtype | str
        The data type of samples from the distribution. (default is `tf.int32`)

    sample_num : int | tf.Tensor | tf.Variable
        Optional number of samples to take. (default None)

    enum_as_samples : bool
        Whether or not to take enumeration "samples" rather than
        random samples? (default False)

        This argument cannot be True when `sample_num` is specified.

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
                 enum_as_samples=False, observed=None,
                 validate_observed_shape=False, group_event_ndims=None,
                 name=None, default_name=None):
        super(Bernoulli, self).__init__(
            distribution=lambda: distributions.Bernoulli(
                logits=logits,
                probs=probs,
                dtype=dtype,
                name='distribution'
            ),
            sample_num=sample_num,
            enum_as_samples=enum_as_samples,
            observed=observed,
            validate_observed_shape=validate_observed_shape,
            group_event_ndims=group_event_ndims,
            name=name,
            default_name=default_name,
        )
