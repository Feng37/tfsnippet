# -*- coding: utf-8 -*-
import tensorflow as tf

__all__ = [
    'get_default_session_or_error',
    'get_variables_as_dict', 'set_variables_from_dict',
]


def get_default_session_or_error():
    """Get the default session, or raise an error if there's no one.

    Raises
    ------
    RuntimeError
        If there's no active session.
    """
    ret = tf.get_default_session()
    if ret is None:
        raise RuntimeError('No session is active.')
    return ret


def get_variables_as_dict(collection=tf.GraphKeys.GLOBAL_VARIABLES,
                          scope=None):
    """Get the values of TensorFlow variables as dict.

    Parameters
    ----------
    collection : str
        Choose variables from this collection. Default is GLOBAL_VARIABLES.

    scope : str
        If specified, will choose the variables whose name matches
        the regex pattern specified in this argument.

    Returns
    -------
    dict[str, np.ndarray]
        The dict from variable names to values.
    """
    session = get_default_session_or_error()
    variables = tf.get_collection(collection, scope)
    values = session.run(variables)
    return {var.name: v for (var, v) in zip(variables, values)}


def set_variables_from_dict(var_dict,
                            collection=tf.GraphKeys.GLOBAL_VARIABLES,
                            scope=None,
                            op_name=None):
    """Set the values of TensorFlow variables according to dict.

    Parameters
    ----------
    var_dict : dict[str, np.ndarray]
        The dict from variable names to values.

    collection : str
        Choose variables from this collection. Default is GLOBAL_VARIABLES.

    scope : str
        If specified, will choose the variables whose name matches
        the regex pattern specified in this argument.

    op_name : str
        Name scope of the assignment operations.

    Raises
    ------
    KeyError
        If any of the chosen variables does not exist in `var_dict`.

        Note that it will not cause an error if auxiliary values are
        specified in `var_dict` without any chosen variable depending
        on these values.
    """
    session = get_default_session_or_error()
    variables = tf.get_collection(collection, scope)
    for v in variables:
        if v.name not in var_dict:
            raise KeyError('The value of variable %r is not specified in '
                           '`var_dict`.' % (v.name,))
    with tf.name_scope(op_name, default_name='set_variables_from_dict'):
        assign_op = [tf.assign(v, var_dict[v.name]) for v in variables]
    session.run(assign_op)
