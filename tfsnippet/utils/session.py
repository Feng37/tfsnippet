# -*- coding: utf-8 -*-
import os

import tensorflow as tf
from logging import getLogger

__all__ = [
    'get_default_session_or_error', 'try_get_variable_value',
    'get_uninitialized_variables', 'ensure_variables_initialized',
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


def get_uninitialized_variables(variables=None):
    """Get uninitialized variables as a list.

    Parameters
    ----------
    variables : collections.Iterable[tf.Variable]
        Return only uninitialized variables within this list.
        If not specified, will return all uninitialized variables.

    Returns
    -------
    list[tf.Variable]
    """
    sess = get_default_session_or_error()
    if variables is None:
        variables = tf.global_variables()
    else:
        variables = list(variables)
    init_flag = sess.run(tf.stack(
        [tf.is_variable_initialized(v) for v in variables]
    ))
    return [v for v, f in zip(variables, init_flag) if not f]


def ensure_variables_initialized(variables=None):
    """Ensure all variables are initialized.

    Parameters
    ----------
    variables : collections.Iterable[tf.Variable]
        Ensure only these variables to be initialized.
        If not specified, will ensure all variables initialized.
    """
    uninitialized = get_uninitialized_variables(variables)
    if uninitialized:
        sess = get_default_session_or_error()
        sess.run(tf.variables_initializer(uninitialized))


class VariableSaver(object):
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

    name : str
        Name of this session restorer.
    """

    def __init__(self, variables, save_dir, max_versions=2,
                 filename='variables.dat', latest_file='latest',
                 save_meta=True, name='SessionRestorer'):
        if not isinstance(variables, dict):
            variables = list(variables)
        if max_versions < 2:
            raise ValueError('At least 2 versions should be kept.')
        self.variables = variables
        self.save_dir = os.path.abspath(save_dir)
        self.filename = filename
        self.max_versions = max_versions
        self.latest_file = latest_file
        self.save_meta = save_meta
        self.name = name
        self._saver = self._build_saver()

    def _build_saver(self):
        return tf.train.Saver(var_list=self.variables,
                              max_to_keep=self.max_versions,
                              name=self.name)

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

    def restore(self):
        """Restore the checkpoint from file if it exists."""
        file_path = self.get_latest_file()
        if file_path:
            sess = get_default_session_or_error()
            getLogger(__name__).info(
                'Restore from checkpoint file %r.',
                file_path
            )
            self._saver.restore(sess, file_path)
