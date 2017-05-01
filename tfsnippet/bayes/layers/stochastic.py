# -*- coding: utf-8 -*-
from tfsnippet import distributions
from .base import StochasticTensor

__all__ = ['Normal']


class Normal(StochasticTensor):
    """Stochastic tensor of normal distribution.
    
    Parameters
    ----------
    mean : tf.Tensor | np.ndarray | float
        The mean of the Normal distribution.
        Should be broadcastable to match `stddev`.

    stddev : tf.Tensor | np.ndarray | float
        The standard derivation of the Normal distribution.
        Should be broadcastable to match `mean`.

    logstd : tf.Tensor | np.ndarray | float
        The log standard derivation of the Normal distribution.
        Should be broadcastable to match `mean`.

        If `stddev` is specified, then `logstd` will be ignored.
        
    sample_num : int | tf.Tensor | tf.Variable
        Optional number of samples to take. (default None)
        
    observed : tf.Tensor
        If specified, random sampling will not take place.
        Instead this tensor will be regarded as sampled tensor.
        (default None)

    group_event_ndims : int
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

    name : str
        Name of this normal distribution.
        
    default_name : str
        Default name of this normal distribution.
    """

    def __init__(self, mean, stddev=None, logstd=None, sample_num=None,
                 observed=None, group_event_ndims=None,
                 name=None, default_name=None):
        super(Normal, self).__init__(
            distribution=lambda: distributions.Normal(
                mean=mean,
                stddev=stddev,
                logstd=logstd,
                name='distribution'
            ),
            sample_num=sample_num,
            observed=observed,
            group_event_ndims=group_event_ndims,
            name=name,
            default_name=default_name,
        )
