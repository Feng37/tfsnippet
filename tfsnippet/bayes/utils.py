# -*- coding: utf-8 -*-
import tensorflow as tf

__all__ = [
    'StochasticObject',
    'gather_log_prob',
    'log_prob_reduce_group_events',
]


class StochasticObject(object):
    """Base interface for stochastic objects.
    
    A stochastic object should be any object in a TensorFlow model,
    which has a normalized or un-normalized log-probability.
    """

    @property
    def is_log_prob_normalized(self):
        """Whether or not the log-probability is normalized?"""
        raise NotImplementedError()

    def log_prob(self, group_event_ndims=None, name=None):
        """Compute the log-probability of this stochastic object.

        Parameters
        ----------
        group_event_ndims : int
            If specify, this number of dimensions at the end of `batch_shape`
            would be considered as a group of events, whose probabilities are
            to be accounted together. (default None)

        name : str
            Optional name of this operation.

        Returns
        -------
        tf.Tensor
            The log-probability of this stochastic object.
        """
        raise NotImplementedError()


def gather_log_prob(tensors, name=None):
    """Gather the log-probability of specified objects.

    Parameters
    ----------
    tensors : collections.Iterable[StochasticObject | tf.Tensor]
        For `StochasticObject`, the `log_prob` method will be called
        to get the log-probability.  Otherwise treated as pre-computed
        log-probability.

    name : str
        Optional name of this operation.

    Returns
    -------
    tuple[tf.Tensor]
        The log-probabilities as tensors.
    """
    with tf.name_scope(name, default_name='local_log_prob', values=tensors):
        return tuple(
            t.log_prob()
            if isinstance(t, StochasticObject) else tf.convert_to_tensor(t)
            for t in tensors
        )


def log_prob_reduce_group_events(log_prob, ndims, name=None):
    """Reduce the dimensions of group events in log-probability.

    Parameters
    ----------
    log_prob : tf.Tensor
        The log-probability tensor.

    ndims : int | tf.Tensor
        The number of dimensions at the tail of `log_prob` to be reduced.

    name : str
        Optional name of this operation.

    Returns
    -------
    tf.Tensor
        The reduced log-probability tensor.
    """
    with tf.name_scope(name=name, default_name='log_prob_reduce_group_events'):
        return tf.reduce_sum(
            log_prob,
            axis=tf.range(-ndims, 0)
        )
