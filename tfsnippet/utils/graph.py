# -*- coding: utf-8 -*-
import tensorflow as tf

__all__ = [
    'get_default_graph_or_error'
]


def get_default_graph_or_error():
    """Get the default graph, or raise an error if there's no one.

    Raises
    ------
    RuntimeError
        If there's no default graph.
    """
    ret = tf.get_default_graph()
    if ret is None:
        raise RuntimeError('No graph is active.')
    return ret
