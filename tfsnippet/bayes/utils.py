# -*- coding: utf-8 -*-
import tensorflow as tf

from .tensors import StochasticTensor

__all__ = ['local_log_prob']


def local_log_prob(tensors, name=None):
    """Compute the conditional log-prob of each stochastic tensor.

    This methods computes the conditional log-probability of each specified
    stochastic tensor, given all its ancestors.  It is like a local detector,
    where only the tensor itself as well as its inputs are available for
    computing the log-prob of each stochastic tensor, thus it is named as
    `local_log_prob`.

    Parameters
    ----------
    tensors : collections.Iterable[StochasticTensor]
        The stochastic tensors whose log-prob should be computed.

    name : str
        Optional name of this operation.

    Returns
    -------
    tuple[tf.Tensor]
        The computed log-prob as tensors.
    """
    tensors = tuple(tensors)
    if not tensors:
        return ()
    for t in tensors:
        if not isinstance(t, StochasticTensor):
            raise TypeError('%r is not a StochasticTensor.' % (t,))
    with tf.name_scope(name, default_name='local_log_prob', values=tensors):
        return tuple(t.log_prob() for t in tensors)
