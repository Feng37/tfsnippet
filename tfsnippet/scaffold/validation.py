# -*- coding: utf-8 -*-
import os
import shutil
from contextlib import contextmanager
from logging import getLogger

import tensorflow as tf
from tfsnippet.utils import VariableSaver, TemporaryDirectory, makedirs

__all__ = ['early_stopping']


class _EarlyStopping(object):
    """Class to hold the best loss within an early-stopping context.

    Parameters
    ----------
    saver : VariableSaver
        The variable saver for early-stopping.

    best_loss : float
        The initial best loss (default None).

    smaller_is_better : bool
        Whether or not the less, the better loss? (default True)
    """

    def __init__(self, saver, best_loss=None, smaller_is_better=True):
        self._saver = saver
        self._best_loss = best_loss
        self._smaller_is_better = smaller_is_better

    def update(self, loss, global_step=None):
        """Update the best loss.

        Parameters
        ----------
        loss : float
            New loss value.

        global_step : int
            Optional global step counter.
            
        Returns
        -------
        bool
            Whether or not the best loss has been updated?
        """
        if self._best_loss is None or \
                (self._smaller_is_better and loss < self._best_loss) or \
                (not self._smaller_is_better and loss > self._best_loss):
            self._saver.save(global_step)
            self._best_loss = loss
            return True
        return False

    @property
    def best_loss(self):
        """Get the current best loss."""
        return self._best_loss


@contextmanager
def early_stopping(param_vars, save_dir=None, smaller_is_better=True,
                   restore_on_error=False, cleanup=True, name=None):
    """Open a context to memorize the values of parameters at best loss.

    This method will open a context with an object to memorize the best
    loss for early-stopping.  An example of using this early-stopping
    context is:

        with early_stopping(param_vars) as es:
            ...
            es.update(loss, global_step)
            ...

    Where ``es.update(loss, global_step)`` should cause the parameters to
    be saved on disk if `loss` is better than the current best loss.
    One may also get the best loss via ``es.best_loss``.

    Parameters
    ----------
    param_vars : list[tf.Variable] | dict[str, tf.Variable]
        List or dict of variables to be memorized.

        If a dict is specified, the keys of the dict would be used as the
        serializations keys via `VariableSaver`.

    save_dir : str
        The directory where to save the variable values.
        If not specified, will use a temporary directory.

    smaller_is_better : bool
        Whether or not the less, the better loss? (default True)

    restore_on_error : bool
        Whether or not to restore the memorized parameters even on error?
        (default False)

    cleanup : bool
        Whether or not to cleanup the saving directory on exit?

        This argument will be ignored if `save_dir` is None, while
        the temporary directory will always be deleted on exit.

    name : str
        Optional name of this scope.

    Yields
    ------
    _EarlyStopping
        The object to receive loss during early-stopping context.
    """
    if save_dir is None:
        with TemporaryDirectory() as tempdir:
            with early_stopping(param_vars, tempdir, cleanup=False,
                                smaller_is_better=smaller_is_better,
                                restore_on_error=restore_on_error,
                                name=name) as es:
                yield es

    else:
        with tf.name_scope(name):
            saver = VariableSaver(param_vars, save_dir)
            save_dir = os.path.abspath(save_dir)
            makedirs(save_dir, exist_ok=True)
            es = _EarlyStopping(saver, smaller_is_better=smaller_is_better)

            try:
                yield es
            except Exception as ex:
                if isinstance(ex, KeyboardInterrupt) or restore_on_error:
                    saver.restore()
                raise
            else:
                saver.restore()
            finally:
                if cleanup:
                    try:
                        if os.path.exists(save_dir):
                            shutil.rmtree(save_dir)
                    except Exception:
                        getLogger(__name__).error(
                            'Failed to cleanup validation save dir %r.',
                            save_dir, exc_info=True
                        )
