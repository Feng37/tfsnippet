# -*- coding: utf-8 -*-
import os

import six
import tensorflow as tf
from logging import getLogger

from .scope import VarScopeObject, lagacy_default_name_arg

__all__ = [
    'get_default_session_or_error', 'try_get_variable_value',
    'get_uninitialized_variables', 'ensure_variables_initialized',
    'get_variable_values', 'set_variable_values',
    'VariableSaver',
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


def try_get_variable_value(v, sess=None):
    """Attempt to get the variable value.

    Parameters
    ----------
    v : tf.Variable
        The variable whose value is to be fetched.

    sess : tf.Session
        The session.  If not specified, use the active session.

    Returns
    -------
    any | None
        Returns the value of the variable if it is initialized,
        otherwise returns None.
    """
    if sess is None:
        sess = get_default_session_or_error()
    try:
        return sess.run(v)
    except tf.errors.FailedPreconditionError:
        return None


def get_uninitialized_variables(variables=None, name=None):
    """Get uninitialized variables as a list.

    Parameters
    ----------
    variables : collections.Iterable[tf.Variable]
        Return only uninitialized variables within this list.
        If not specified, will return all uninitialized global variables.

    name : str
        Optional name of this operation.

    Returns
    -------
    list[tf.Variable]
    """
    sess = get_default_session_or_error()
    if variables is None:
        variables = tf.global_variables()
    else:
        variables = list(variables)
    with tf.name_scope(name, default_name='get_uninitialized_variables'):
        init_flag = sess.run(tf.stack(
            [tf.is_variable_initialized(v) for v in variables]
        ))
    return [v for v, f in zip(variables, init_flag) if not f]


def ensure_variables_initialized(variables=None, name=None):
    """Ensure all variables are initialized.

    Parameters
    ----------
    variables : collections.Iterable[tf.Variable]
        Ensure only these variables to be initialized.
        If not specified, will ensure all variables initialized.

    name : str
        Optional name of this operation.
    """
    with tf.name_scope(name, default_name='ensure_variables_initialized'):
        uninitialized = get_uninitialized_variables(variables)
        if uninitialized:
            sess = get_default_session_or_error()
            sess.run(tf.variables_initializer(uninitialized))


def get_variable_values(variables):
    """Get the values of variables.

    Parameters
    ----------
    variables : list[tf.Variable] | dict[str, tf.Variable]
        A list or a dict of variables.

    Returns
    -------
    list[any] | dict[str, any]
        The value list if `variables` are specified as a list,
        or dict if `variables` are specified as a dict.
    """
    session = get_default_session_or_error()
    if isinstance(variables, dict):
        names = list(variables.keys())
        values = session.run([variables[n] for n in names])
        return {n: v for n, v in zip(names, values)}
    else:
        return session.run(list(variables))


def set_variable_values(variables, values, name=None):
    """Set the values of variables.

    Note that this method will create a new set of graph nodes each time
    it is called, thus is not suitable for repeated variable assignments.

    Parameters
    ----------
    variables : list[tf.Variable] | dict[str, tf.Variable]
        A list or a dict of variables.

    values : list[any] | dict[str, any]
        The list or the dict of values.

    name : str
        Optional name of this operation.

    Raises
    ------
    KeyError | IndexError
        If the specified values does not match variables.
    """
    # collect the variables and the values
    assign_dict = {}
    if isinstance(variables, dict):
        if not isinstance(values, dict):
            raise TypeError('`values` is expected to be a dict '
                            'since `variables` is a dict.')
        for name, var in six.iteritems(variables):
            if not isinstance(var, tf.Variable):
                raise TypeError('%r is not a variable.' % (var,))
            assign_dict[var] = values[name]

    else:
        variables = list(variables)
        values = list(values)
        if len(variables) != len(values):
            raise IndexError('Length of `values` does not match `variables`.')
        for var, value in zip(variables, values):
            if not isinstance(var, tf.Variable):
                raise TypeError('%r is not a variable.' % (var,))
            assign_dict[var] = value

    # get the session
    session = get_default_session_or_error()

    # perform assign operations
    with tf.name_scope(name, default_name='set_variable_values'):
        session.run([
            tf.assign(var, value)
            for var, value in six.iteritems(assign_dict)
        ])


class VariableSaver(VarScopeObject):
    """Version controlled saving and restoring TensorFlow variables.

    Parameters
    ----------
    variables : collections.Iterable[tf.Variable] | dict[str, any]
        List of variables, or dict of variables with explicit keys,
        which should be saved and restored.

    save_dir : str
        Directory where to place the saved variables.

    max_versions : int
        Maximum versions to keep in the directory (Default is 2).

        At least 2 versions should be kept, in order to prevent corrupted
        checkpoint files caused by IO failure.

    filename : str
        Name of the files of variable values (default is "variables.dat").

    latest_file : str
        Name of the file which organizes the checkpoint versions
        (default is "latest").

    save_meta : bool
        Whether or not to save meta graph (default is True).

    name, scope : str
        Optional name and scope of this session restorer.
    """

    @lagacy_default_name_arg
    def __init__(self, variables, save_dir, max_versions=2,
                 filename='variables.dat', latest_file='latest',
                 save_meta=True, name=None, scope=None):
        if not isinstance(variables, dict):
            variables = list(variables)
        if max_versions < 2:
            raise ValueError('At least 2 versions should be kept.')
        super(VariableSaver, self).__init__(scope, name)
        self.variables = variables
        self.save_dir = os.path.abspath(save_dir)
        self.filename = filename
        self.max_versions = max_versions
        self.latest_file = latest_file
        self.save_meta = save_meta
        with tf.variable_scope(self.variable_scope):
            self._saver = tf.train.Saver(
                var_list=self.variables, max_to_keep=self.max_versions,
                name='saver'
            )

    def get_latest_file(self):
        """Get the latest available checkpoint file."""
        return tf.train.latest_checkpoint(self.save_dir, self.latest_file)

    def save(self, global_step=None):
        """Save the checkpoint to file.

        Parameters
        ----------
        global_step : int | tf.Tensor
            The global step counter.
        """
        sess = get_default_session_or_error()
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self._saver.save(
            sess,
            os.path.join(self.save_dir, self.filename),
            global_step=global_step,
            latest_filename=self.latest_file,
            write_meta_graph=self.save_meta
        )

    def restore(self, ignore_non_exist=True):
        """Restore the checkpoint from file if it exists.

        Parameters
        ----------
        ignore_non_exist : bool
            Whether or not to ignore error if the saved file does not exist?
            (default True)
        """
        file_path = self.get_latest_file()
        if file_path:
            sess = get_default_session_or_error()
            self._saver.restore(sess, file_path)
            getLogger(__name__).debug(
                'Restored from checkpoint file %r.', file_path)
        elif not ignore_non_exist:
            raise IOError(
                'Checkpoint file does not exist at directory %r.' %
                (self.save_dir,)
            )
