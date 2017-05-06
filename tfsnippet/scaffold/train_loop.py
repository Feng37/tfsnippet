# -*- coding: utf-8 -*-
import os
import time
from collections import OrderedDict
from contextlib import contextmanager

import tensorflow as tf

from .logging import MetricLogger, SummaryWriter, get_parameters_summary, \
    MetricFormatter

__all__ = ['train_loop']


def _print_function(message):
    """Default print function, that outputs `message` to stdout."""
    print(message)


class _TrainLoop(object):
    """Training loop context object.

    Parameters
    ----------
    param_vars : list[tf.Variable] | dict[str, tf.Variable]
        List or dict of variables, which should be optimized during training.

    print_function : (str) -> None
        Function to print the log messages (default is equivalent to `print`)

        An alternative of this argument may be ``getLogger(__name__).info``,
        such that the log messages will be printed via logging facilities.

    metric_formatter : MetricFormatter
        The formatter of the training metrics.

    summary_dir : str
        Directory to write TensorFlow summaries.
        Ignored if `summary_writer` is specified.

    summary_writer : SummaryWriter | tf.summary.FileWriter
        TensorFlow summary writer for writing metrics onto disk.

    initial_epoch, initial_step : int | tf.Tensor | tf.Variable
        The initial epoch and step counter.
    """

    def __init__(self,
                 param_vars,
                 print_function,
                 metric_formatter,
                 summary_dir=None,
                 summary_writer=None,
                 initial_epoch=0,
                 initial_step=0):
        if not isinstance(param_vars, (dict, OrderedDict)):
            param_vars = tuple(param_vars)

        if isinstance(initial_epoch, (tf.Variable, tf.Tensor)):
            initial_epoch = int(initial_epoch.eval())
        if isinstance(initial_step, (tf.Variable, tf.Tensor)):
            initial_step = int(initial_step.eval())

        close_summary_writer = False
        if isinstance(summary_writer, tf.summary.FileWriter):
            summary_writer = SummaryWriter(summary_writer)
        elif summary_writer is None:
            if summary_dir is not None:
                summary_dir = os.path.abspath(summary_dir)
                summary_writer = SummaryWriter(
                    tf.summary.FileWriter(summary_dir))
                close_summary_writer = True
        elif not isinstance(summary_writer, SummaryWriter):
            raise TypeError(
                '`summary_writer` is expected to be `SummaryWriter`, but '
                'got %r.' % (summary_writer,)
            )

        self._param_vars = param_vars
        self._print_function = print_function
        self._metric_formatter = metric_formatter
        self._summary_writer = summary_writer   # type: SummaryWriter
        self._close_summary_writer = close_summary_writer

        # metric accumulators

        # epoch and step counter
        self._epoch = initial_epoch  # type: int
        self._step = initial_step  # type: int
        self._max_epoch = None
        self._max_step = None

    @property
    def param_vars(self):
        """Get the trainable parameter variables."""
        return self._param_vars

    @property
    def epoch(self):
        """Get the epoch counter (starting from 1)."""
        return self._epoch

    @property
    def step(self):
        """Get the global step counter (starting from 1)."""
        return self._step

    def iter_epochs(self):
        """Iterate through the epochs.

        Yields
        ------
        int
            The epoch counter (starting from 1).
        """
        try:
            while self._max_epoch is None or self._epoch < self._max_epoch:
                self._epoch += 1
                start_time = time.time()
                yield self._epoch

                # add metrics of this epoch
                self._summary_writer.add_metrics(
                    self._step,
                    epoch_time=time.time() - start_time
                )
        finally:
            pass

    def iter_steps(self, data_generator=None):
        """Iterate through the steps.

        Parameters
        ----------
        data_generator
            Optional iterable data to be yielded at every step.

        Yields
        ------
        int | (int, any)
            The step counter (starting from 1), or the step and data
            if `data_generator` is specified.
        """
        try:
            if data_generator is not None:
                data_generator = iter(data_generator)

            while self._max_step is None or self._step < self._max_step:
                # prepare for the step data
                if data_generator is None:
                    yield_obj = self._step
                else:
                    try:
                        step_data = next(data_generator)
                    except StopIteration:
                        break
                    yield_obj = self._step, step_data

                # yield this step
                self._step += 1
                start_time = time.time()
                yield yield_obj

                # add metrics of this step
                self._summary_writer.add_metrics(
                    self._step,
                    step_time=time.time() - start_time
                )
        finally:
            pass

    @property
    def summary_writer(self):
        """Get the summary writer."""
        return self._summary_writer

    def println(self, message):
        """Print `message` via `print_function`."""
        self._print_function(message)

    def print_training_summary(self):
        """Print the training summary.

        The training summary should include:

        *   Execution environment.
        *   Parameter variables to be optimized during training.
        """
        self.println(get_parameters_summary(variables=self._param_vars))
        self.println('')

    def close(self):
        """Cleanup the training loop."""
        if self._close_summary_writer:
            self._summary_writer.close()
            self._close_summary_writer = False


@contextmanager
def train_loop(param_vars,
               print_function=_print_function,
               metric_formatter=MetricFormatter(),
               summary_dir=None,
               summary_writer=None,
               initial_epoch=0,
               initial_step=0):
    """Open a training loop context.

    This method should open a context for training loop, and provide an object
    shipped with a set of convenient methods for common routines in training.
    Basically, it should be useful for maintaining epoch and step counters,
    logging training metrics, memorizing best parameters via early-stopping,
    and many more.

    Parameters
    ----------
    param_vars : list[tf.Variable] | dict[str, tf.Variable]
        List or dict of variables, which should be optimized during training.

    print_function : (str) -> None
        Function to print the log messages (default is equivalent to `print`)

        A common choice of this argument may be ``getLogger(__name__).info``,
        such that the log messages will be printed via logging facilities.

    metric_formatter : MetricFormatter
        The formatter of the training metrics.
        If not specified, an instance of `MetricFormatter` will be used.

    summary_dir : str
        Directory to write TensorFlow summaries.
        Ignored if `summary_writer` is specified.

    summary_writer : SummaryWriter | tf.summary.FileWriter
        TensorFlow summary writer for writing metrics onto disk.

    initial_epoch, initial_step : int | tf.Tensor | tf.Variable
        The initial epoch and step counter (default 0).

        Value of the initial counter should be one less than the actual
        first epoch or step to be taken. (i.e., "0" instead of "1")

    Yields
    ------
    _TrainLoop
        The training loop context object.
    """
    loop = _TrainLoop(
        param_vars=param_vars,
        print_function=print_function,
        metric_formatter=metric_formatter,
        summary_dir=summary_dir,
        summary_writer=summary_writer,
        initial_epoch=initial_epoch,
        initial_step=initial_step,
    )
    try:
        yield loop
    finally:
        loop.close()
