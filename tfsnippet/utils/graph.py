# -*- coding: utf-8 -*-
import tensorflow as tf

__all__ = [
    'get_default_graph_or_error', 'get_global_step',
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


def get_global_step(graph=None):
    """Get the global step tensor from `graph`.

    If the `graph` is not specified, use the default graph.
    Furthermore, if the global step tensor does not exist,
    create a new one of int64 type.

    Parameters
    ----------
    graph : tf.Graph
        The graph, and if not specified, use the default graph.

    Returns
    -------
    tf.Variable
        The global step tensor.
    """
    graph = graph or get_default_graph_or_error()
    global_step = tf.train.get_global_step(graph)
    if global_step is None:
        global_step = tf.get_variable(
            'global_step',
            initializer=0,
            dtype=tf.int64,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                         tf.GraphKeys.GLOBAL_STEP],
            trainable=False
        )
    return global_step
