# -*- coding: utf-8 -*-
import os

import tensorflow as tf
from logging import getLogger

__all__ = [
    'get_default_session_or_error', 'try_get_variable_value',
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
