# -*- coding: utf-8 -*-
import tensorflow as tf

__all__ = [
    'StochasticObject',
    'gather_log_lower_bound',
    'reduce_log_lower_bound',
]


class StochasticObject(object):
    """Base interface for stochastic objects.

    A stochastic object should be any object in a TensorFlow model,
    which has a log-probability lower-bound.
    """

    def log_lower_bound(self, group_event_ndims=None, name=None):
        """Compute the log-probability lower-bound.

        Parameters
        ----------
        group_event_ndims : int
            If specify, this number of dimensions at the end of `batch_shape`
            would be considered as a group of events, whose log-probability
            lower-bounds are summed together. (default None)

        name : str
            Optional name of this operation.

        Returns
        -------
        tf.Tensor
            The log-probability lower-bound.
        """
        raise NotImplementedError()


def gather_log_lower_bound(tensors, name=None):
    """Gather the log-probability lower-bounds of specified objects.

    Parameters
    ----------
    tensors : collections.Iterable[StochasticObject | tf.Tensor]
        For `StochasticObject`, the `log_lower_bound` method will be
        called to get the lower-bound.  Otherwise treated as pre-computed
        lower-bounds.

    name : str
        Optional name of this operation.

    Returns
    -------
    tuple[tf.Tensor]
        The log-probabilities as tensors.
    """
    with tf.name_scope(name, default_name='local_log_prob', values=tensors):
        return tuple(
            t.log_lower_bound()
            if isinstance(t, StochasticObject) else tf.convert_to_tensor(t)
            for t in tensors
        )


def reduce_log_lower_bound(log_prob, ndims, name=None):
    """Reduce the dimensions of group events in log-probability lower-bounds.

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
