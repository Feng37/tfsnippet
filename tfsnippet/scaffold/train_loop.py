# -*- coding: utf-8 -*-
import os
import time
from collections import OrderedDict
from contextlib import contextmanager

import six
import tensorflow as tf

from tfsnippet.utils import MetricAccumulator
from .logging import (SummaryWriter, get_variables_summary, MetricFormatter,
                      MetricLogger)
from .validation import _EarlyStopping, early_stopping as open_early_stopping

__all__ = [
    'train_loop', '_TrainLoop',
]

_EPOCH_TIME_METRIC = 'epoch_time'
_STEP_TIME_METRIC = 'step_time'


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

    summary_writer : SummaryWriter
        TensorFlow summary writer for writing metrics onto disk.

    metric_formatter : MetricFormatter
        The formatter of the training metrics.

    valid_metric : str
        Name of the validation metric.

    initial_valid_metric : float
        Initial value of the validation metric.

    valid_metric_smaller_is_better : bool
        Whether or not the smaller validation metric is better?

    early_stopping : _EarlyStopping
        The early-stopping context.

    initial_epoch, initial_step : int
        The initial epoch and step counter.

    max_epoch, max_step : int
        The configured maximum values for epoch and step counter.
    """

    def __init__(self,
                 param_vars,
                 print_function,
                 summary_writer,
                 metric_formatter,
                 valid_metric,
                 initial_valid_metric,
                 valid_metric_smaller_is_better,
                 early_stopping,
                 initial_epoch,
                 initial_step,
                 max_epoch,
                 max_step):
        self._param_vars = param_vars
        self._print_function = print_function
        self._metric_formatter = metric_formatter
        self._early_stopping = early_stopping
        self._valid_metric = valid_metric
        self._valid_metric_smaller_is_better = valid_metric_smaller_is_better
        self._summary_writer = summary_writer   # type: SummaryWriter

        # metric accumulators
        self._step_metrics = MetricLogger(self._metric_formatter)
        self._epoch_metrics = MetricLogger(self._metric_formatter)

        # flag to track the context
        self._within_epoch = False
        self._within_step = False

        # epoch and step counter
        self._epoch = initial_epoch
        self._step = initial_step
        self._max_epoch = max_epoch
        self._max_step = max_step

        # epoch and step context flags
        self._best_valid_metric = initial_valid_metric
        self._is_best_valid_metric = False
        self._epoch_start_time = None
        self._step_start_time = None

    def _commit_epoch_start_time(self):
        if self._epoch_start_time is not None:
            duration = time.time() - self._epoch_start_time
            self.add_metrics(metrics={_EPOCH_TIME_METRIC: duration})
            self._epoch_start_time = None

    def _commit_step_start_time(self):
        if self._step_start_time is not None:
            duration = time.time() - self._step_start_time
            self.add_metrics(metrics={_STEP_TIME_METRIC: duration})
            self._step_start_time = None

    @property
    def summary_writer(self):
        """Get the summary writer instance."""
        return self._summary_writer

    @property
    def param_vars(self):
        """Get the trainable parameter variables."""
        return self._param_vars

    @property
    def epoch(self):
        """Get the epoch counter (starting from 1)."""
        return self._epoch

    @property
    def max_epoch(self):
        """Get or set the max value for epoch counter."""
        return self._max_epoch

    @max_epoch.setter
    def max_epoch(self, value):
        self._max_epoch = int(value)

    @property
    def step(self):
        """Get the global step counter (starting from 1)."""
        return self._step

    @property
    def max_step(self):
        """Get or set the max value for global step counter."""
        return self._max_step

    @max_step.setter
    def max_step(self, value):
        self._max_step = int(value)

    @property
    def best_valid_metric(self):
        """Get the best valid metric."""
        return self._best_valid_metric

    @property
    def summary_writer(self):
        """Get the summary writer."""
        return self._summary_writer

    def iter_epochs(self):
        """Iterate through the epochs.

        This method can only be called when there's no other epoch loop
        is being iterated.  Furthermore, after exiting this loop, both
        the epoch metrics as well as the step metrics will be cleared.

        Yields
        ------
        int
            The epoch counter (starting from 1).

            If `max_epoch` is configured, it will stop at
        """
        def loop_condition():
            return (
                (self._max_epoch is None or self._epoch < self._max_epoch) and
                (self._max_step is None or self._step < self._max_step)
            )

        if self._within_epoch:
            raise RuntimeError('Another epoch loop has been opened.')
        try:
            while loop_condition():
                self._epoch += 1
                self._within_epoch = True
                self._epoch_start_time = time.time()
                yield self._epoch
                self._commit_epoch_start_time()
        finally:
            self._within_epoch = False
            self._epoch_start_time = None
            self._step_metrics.clear()
            self._epoch_metrics.clear()
            self._is_best_valid_metric = False

    def iter_steps(self, data_generator=None):
        """Iterate through the steps.

        This method can only be called when there's no other step loop
        is being iterated, and an epoch loop is active.

        Parameters
        ----------
        data_generator
            Optional iterable data to be yielded at every step.

            This is required if `max_step` is not configured, so as to
            prevent an unstoppable step loop.

        Yields
        ------
        int | (int, any)
            The global step counter (starting from 1), or the step
            and data if `data_generator` is specified.
        """
        def loop_condition():
            return self._max_step is None or self._step < self._max_step

        if not self._within_epoch:
            raise RuntimeError('Step loop must be opened within active epoch '
                               'loop.')
        if self._within_step:
            raise RuntimeError('Another step loop has been opened.')
        if self._max_step is None and data_generator is None:
            raise RuntimeError('`data_generator` is required when `max_step` '
                               'is not configured, so as to prevent an '
                               'unstoppable step loop.')

        try:
            if data_generator is not None:
                data_generator = iter(data_generator)

            while loop_condition():
                # prepare for the step data
                if data_generator is None:
                    yield_obj = self._step + 1
                else:
                    try:
                        step_data = next(data_generator)
                    except StopIteration:
                        break
                    yield_obj = self._step + 1, step_data

                # yield this step
                self._step += 1
                self._within_step = True
                self._step_start_time = time.time()
                yield yield_obj
                self._commit_step_start_time()
        finally:
            self._within_step = False
            self._step_start_time = None

    def _require_context(self):
        if not self._within_epoch and not self._within_step:
            raise RuntimeError(
                'Neither an epoch nor a step loop has been opened.')

    @contextmanager
    def timeit(self, metric_name):
        """Open a context for timing.

        Parameters
        ----------
        metric_name : str
            Store the timing result in this metric.

            Note that `metric_name` must end with "time" or "timer",
            otherwise by default the time values will not be formatted
            as human readable time durations.
        """
        self._require_context()
        start_time = time.time()
        yield
        duration = time.time() - start_time
        self.add_metrics(metrics={metric_name: duration})

    @contextmanager
    def accumulator(self, metric_name):
        """Get an accumulator for specified metric.

        The average value of the accumulator will be added to train loop
        after exiting the context.  Other statistics will be discarded.

        Parameters
        ----------
        metric_name : str
            The metric name.

        Yields
        ------
        MetricAccumulator
            The accumulator for metric values.
        """
        self._require_context()
        acc = MetricAccumulator()
        yield acc
        if acc.has_value:
            self.add_metrics(metrics={metric_name: acc.mean})

    def add_metrics(self, metrics=None, **kwargs):
        """Add metric values.

        This method must be called when there's at least an active epoch
        loop.  It will add metrics to the epoch metrics collector, and if
        there's an active step loop, it will also add metrics to the step
        metrics collector.

        If `summary_writer` is configured, it will also write the metrics
        as summaries onto disk.  Furthermore, if `early_stopping_metric`
        is configured, it will also perform early-stopping.

        Parameters
        ----------
        metrics, **kwargs
            Metric values at current step or epoch.
        """
        self._require_context()
        if metrics is not None and not isinstance(metrics, dict):
            raise TypeError('`metrics` should be a dict.')

        if self._within_epoch:
            self._epoch_metrics.add_metrics(metrics, **kwargs)
        if self._within_step:
            self._step_metrics.add_metrics(metrics, **kwargs)
        if self._summary_writer:
            self._summary_writer.add_metrics(self.step, metrics, **kwargs)

        def update_valid_metric(d):
            v = d.get(self._valid_metric)
            if v is not None:
                if self._best_valid_metric is None or \
                        (self._valid_metric_smaller_is_better and
                         v < self._best_valid_metric) or \
                        (not self._valid_metric_smaller_is_better and
                         v > self._best_valid_metric):
                    self._best_valid_metric = v
                    self._is_best_valid_metric = True
                else:
                    self._is_best_valid_metric = False
                if self._early_stopping:
                     self._early_stopping.update(v, self.step)
        if self._valid_metric:
            if metrics:
                update_valid_metric(metrics)
            if kwargs:
                update_valid_metric(kwargs)

    def add_summary(self, summary):
        """Add a summary object.

        Parameters
        ----------
        summary : bytes | tf.summary.Summary
            The summary object.
        """
        self._summary_writer.add_summary(summary, global_step=self.step)

    def println(self, message, with_tag=False):
        """Print `message` via `print_function`."""
        if with_tag:
            def format_tag(v, max_v, name):
                if max_v is not None:
                    return '%s %d/%d' % (name, v, max_v)
                return '%s %d' % (name, v)

            if self._within_step:
                tag = '%s, %s' % (
                    format_tag(self._epoch, self._max_epoch, 'Epoch'),
                    format_tag(self._step, self._max_step, 'Step'),
                )
            elif self._within_epoch:
                tag = format_tag(self._epoch, self._max_epoch, 'Epoch')
            else:
                self._require_context()
            message = '[%s] %s' % (tag, message)
        self._print_function(message)

    def print_training_summary(self):
        """Print the training summary.

        The training summary should include:

        *   Execution environment.
        *   Parameter variables to be optimized during training.
        """
        self.println(get_variables_summary(variables=self._param_vars,
                                           title='Trainable Parameters'))
        self.println('')

    def print_logs(self):
        """Print the training logs.

        This method will print the accumulated metrics.  If there's an
        active step loop, it will print metrics from the step metrics
        collector.  Otherwise if there's only an epoch loop, it will
        print metrics from the epoch metrics accumulator.

        The metrics of corresponding loop context will be cleared after
        the logs are printed.  Besides, the epoch or step timer will be
        committed as metric immediately when this method is called.
        So it must be called at the end of an epoch or a step.
        """
        if self._within_step:
            self._commit_step_start_time()
            metrics = self._step_metrics
        elif self._within_epoch:
            self._commit_epoch_start_time()
            metrics = self._epoch_metrics
        else:
            self._require_context()

        best_mark = ' (*)' if self._is_best_valid_metric else ''
        self.println('%s%s' % (metrics.format_logs(), best_mark),
                     with_tag=True)
        self._is_best_valid_metric = False
        metrics.clear()


@contextmanager
def train_loop(param_vars,
               print_function=_print_function,
               summary_dir=None,
               summary_writer=None,
               metric_formatter=MetricFormatter(),
               valid_metric='valid_loss',
               initial_valid_metric=None,
               early_stopping=False,
               initial_epoch=0,
               initial_step=0,
               max_epoch=None,
               max_step=None):
    """Open a training loop context.

    This method should open a context for training loop, and provide an object
    shipped with a set of convenient methods for common routines in training.
    Basically, it should be useful for maintaining epoch and step counters,
    logging training metrics, memorizing best parameters via early-stopping,
    and many more.  An example of using the `train_loop`:

        with train_loop(param_vars, max_epoch=10, early_stopping=True) as loop:
            loop.print_training_summary()
            for epoch in loop.iter_epochs():
                data_iterator = zip(
                    minibatch_iterator(data_x, batch_size),
                    minibatch_iterator(data_y, batch_size),
                )
                for step, (x, y) in loop.iter_steps(data_iterator):
                    step_loss = session.run(
                        [loss, train_op],
                        feed_dict={input_x: x, input_y: y}
                    )
                    loop.add_metrics(loss=step_loss)
                with loop.timeit('valid_time'):
                    valid_loss = session.run(
                        loss, feed_dict={input_x: test_x, input_y: test_y})
                    loop.add_metrics(valid_loss=valid_loss)
                loop.print_logs()

    Parameters
    ----------
    param_vars : list[tf.Variable] | dict[str, tf.Variable]
        List or dict of variables, which should be optimized during training.

    print_function : (str) -> None
        Function to print the log messages (default is equivalent to `print`)

        A common choice of this argument may be ``getLogger(__name__).info``,
        such that the log messages will be printed via logging facilities.

    summary_dir : str
        Directory to write TensorFlow summaries.
        Ignored if `summary_writer` is specified.

    summary_writer : SummaryWriter | tf.summary.FileWriter
        TensorFlow summary writer for writing metrics onto disk.

    metric_formatter : MetricFormatter
        The formatter of the training metrics.
        If not specified, an instance of `MetricFormatter` will be used.

    valid_metric : str | (str, bool)
        Which metric to use for validation? (default 'valid_loss')

        If specified, the best values of the metric will be monitored.
        A str or a tuple of (str, bool) is expected:

        *   If it is a str, it should be the name of the validation metric.
        *   Otherwise it should be the name of metric, and a flag indicating
            whether or not smaller values are better, given this metric.

        If the flag, whether or not smaller-is-better, is not specified, it
        will be inferred from the metric name.  Given a metric name with
        `acc` or `accuracy` as suffix will imply a larger-is-better metric,
        otherwise the metric will be regarded as smaller-is-better.

    initial_valid_metric : float | tf.Tensor | tf.Variable
        The initial value of the metric used for early-stopping.

    early_stopping : bool
        Whether or not to do early-stopping? (default False)

        If True, early-stopping will be applied on `param_vars`, according
        to the `valid_metric`.

    initial_epoch, initial_step : int | tf.Tensor | tf.Variable
        The initial epoch and step counter (default 0).

        Value of the initial counter should be one less than the actual
        first epoch or step to be taken. (i.e., "0" instead of "1")

    max_epoch, max_step : int | tf.Tensor | tf.Variable
        The configured maximum values for epoch and step counter.

    Yields
    ------
    _TrainLoop
        The training loop context object.
    """
    if not isinstance(param_vars, (dict, OrderedDict)):
        param_vars = list(param_vars)

    if isinstance(initial_valid_metric, (tf.Variable, tf.Tensor)):
        initial_valid_metric = initial_valid_metric.eval()
    if isinstance(initial_epoch, (tf.Variable, tf.Tensor)):
        initial_epoch = int(initial_epoch.eval())
    if isinstance(initial_step, (tf.Variable, tf.Tensor)):
        initial_step = int(initial_step.eval())
    if isinstance(max_epoch, (tf.Variable, tf.Tensor)):
        max_epoch = int(max_epoch.eval())
    if isinstance(max_step, (tf.Variable, tf.Tensor)):
        max_step = int(max_step.eval())

    smaller_is_better = True
    if valid_metric:
        if isinstance(valid_metric, six.string_types):
            valid_metric = valid_metric
            smaller_is_better = not (
                valid_metric.endswith('acc') or
                valid_metric.endswith('accuracy')
            )
        else:
            valid_metric, smaller_is_better = valid_metric

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

    try:
        loop = _TrainLoop(
            param_vars=param_vars,
            print_function=print_function,
            metric_formatter=metric_formatter,
            valid_metric=valid_metric,
            initial_valid_metric=initial_valid_metric,
            valid_metric_smaller_is_better=smaller_is_better,
            early_stopping=None,
            summary_writer=summary_writer,
            initial_epoch=initial_epoch,
            initial_step=initial_step,
            max_epoch=max_epoch,
            max_step=max_step,
        )
        if early_stopping and len(param_vars) > 0:
            with open_early_stopping(param_vars,
                                     initial_metric=initial_valid_metric,
                                     smaller_is_better=smaller_is_better) as es:
                loop._early_stopping = es
                yield loop
        else:
            yield loop
    finally:
        if close_summary_writer:
            summary_writer.close()
