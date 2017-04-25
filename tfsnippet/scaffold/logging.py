# -*- coding: utf-8 -*-
import os
import sys
import time
from collections import defaultdict, OrderedDict
from contextlib import contextmanager

import six
import tensorflow as tf

from tfsnippet.utils import (get_default_session_or_error,
                             ScopedObject,
                             open_variable_scope)

__all__ = [
    'TrainLogger',
]

EPOCH_TIME_METRIC = 'epoch_timer'
STEP_TIME_METRIC = 'step_timer'
VALID_TIME_METRIC = 'valid_timer'
LOSS_METRIC = 'loss'
TRAIN_LOSS_METRIC = 'train_loss'
VALID_LOSS_METRIC = 'valid_loss'


class _MetricAccumulator(object):
    """Accumulator to compute the average of certain metric."""

    def __init__(self, initial_value=0., initial_counter=0):
        if initial_counter < 0:
            raise ValueError('`initial_counter` must be greater or equal to 0.')
        self.initial_value = initial_value
        self.initial_counter = initial_counter
        self.value = initial_value
        self.counter = initial_counter

    @property
    def has_value(self):
        return self.counter > self.initial_counter

    def get_value(self):
        return None if not self.has_value else self.value

    def reset(self):
        self.value = self.initial_value
        self.counter = self.initial_counter

    def submit(self, metric):
        self.counter += 1
        self.value += (metric - self.value) / float(self.counter)


def _print_function(message):
    sys.stdout.write(message + '\n')
    sys.stdout.flush()


def _humanize_time(seconds):
    pieces = []
    for uvalue, uname in [(86400, 'day'),
                          (3600, 'hr'),
                          (60, 'min')]:
        if seconds > uvalue:
            val = int(seconds // uvalue)
            if val > 1:
                uname += 's'
            pieces.append('%d %s' % (val, uname))
            seconds %= uvalue
    pieces.append('%.4g sec%s' % (seconds, 's' if seconds > 1 else ''))
    return ' '.join(pieces)


def _get_tensor_value(v):
    if isinstance(v, (tf.Tensor, tf.Variable)):
        v = get_default_session_or_error().run(v)
    return v


class TrainLogger(ScopedObject):
    """Loss and variables logger.
    
    This class provides convenient methods for printing training losses,
    and for writing variable summaries onto disk.
     
    Parameters
    ----------
    initial_epoch : int | tf.Tensor | tf.Variable
        Specify the initial epoch of the training process. (default 0)
        
    initial_step : int | tf.Tensor | tf.Variable
        Specify the initial step of the training process. (default 0)
        
    max_epoch : int
        Specify the maximum epoch of the training process. (default None)
        
    max_step : int
        Specify the maximum step of the training process. (default None)
        
    metric_format : str
        String format for various metrics. (default '%.6g')
        
    metric_name_width : int
        Maximum width of the formatted metric name. 
        
    compact_log : bool
        Whether or not to print log messages in compact mode?
        
    print_function : (message) -> None
        Function to print the summary messages.
        
    summary_dir : str
        Path of the TensorFlow summary directory.
        If not specified, no summary will be written.
        
    summary_log_timers : bool
        If set to True, will log epoch and step timers in TensorFlow summary.
        Default is False.
    
    name : str
        Name of this TrainLogger.
        
    default_name : str
        Default name of this TrainLogger.
    """

    def __init__(self,
                 initial_epoch=0,
                 initial_step=0,
                 max_epoch=None,
                 max_step=None,
                 metric_format='%.6g',
                 metric_name_width=8,
                 compact_log=False,
                 print_function=_print_function,
                 summary_dir=None,
                 summary_log_timers=False,
                 name=None,
                 default_name=None):
        # check the arguments
        initial_epoch = int(_get_tensor_value(initial_epoch))
        initial_step = int(_get_tensor_value(initial_step))
        if summary_dir is not None:
            summary_dir = os.path.abspath(summary_dir)

        # initialize the attributes
        super(TrainLogger, self).__init__(name, default_name)
        self.epoch = initial_epoch  # type: int
        self.step = initial_step    # type: int
        self.max_epoch = max_epoch
        self.max_step = max_step
        self.metric_format = metric_format
        self.metric_name_width = metric_name_width
        self.compact_log = compact_log
        self.print_function = print_function
        self.summary_dir = summary_dir
        self.summary_log_timers = summary_log_timers

        # the summary writer
        self._summary_writer = None

        # accumulators for various metrics
        self._metrics = defaultdict(_MetricAccumulator)

        # timer for epoch and step
        self._epoch_start_time = None
        self._step_start_time = None
        self._within_epoch = False
        self._within_step = False

    @contextmanager
    def open_summary_writer(self):
        """Open a context equipped with TensorFlow summary writer."""
        if self.summary_dir:
            with open_variable_scope(self.variable_scope):
                self._summary_writer = tf.summary.FileWriter(self.summary_dir)
        try:
            yield
        finally:
            if self._summary_writer:
                self._summary_writer.close()
                self._summary_writer = None

    @property
    def step_for_metrics(self):
        """Get the step counter for current metrics.
        
        If no step context is open, returns `self.step - 1`.  Otherwise 
        returns `self.step`.  This corrected step counter should correspond
        to current metrics.
        """
        return self.step if self._within_step else self.step - 1

    @property
    def epoch_for_metrics(self):
        """Get the epoch counter for current metrics.
        
        If no epoch context is open, returns `self.epoch - 1`.  Otherwise
        returns `self.epoch`.  This corrected epoch counter should correspond
        to current metrics.
        """
        return self.epoch if self._within_epoch else self.epoch - 1

    def get_metric(self, name):
        """Get the accumulated average metric value.
        
        Parameters
        ----------
        name : str
            Name of the metric.
            
        Returns
        -------
        float | None
            Average value of the metric, or None if the metric does not exist.             
        """
        if name in self._metrics:
            return self._metrics[name].get_value()

    def _add_metrics(self, metrics):
        for k, v in metrics:
            self._metrics[k].submit(v)

    def add_metrics(self, metrics):
        """Add one-step metric values.
        
        This method will compose TensorFlow summary objects from
        these metrics, and write to disk with these summaries.
        
        Parameters
        ----------
        metrics : dict[str, float]
            One-step metric values.
        """
        self._add_metrics(six.iteritems(metrics))
        if self._summary_writer:
            # create summaries from metric values
            summary_values = []
            for name, value in six.iteritems(metrics):
                if self.summary_log_timers or name not in (EPOCH_TIME_METRIC,
                                                           STEP_TIME_METRIC,
                                                           VALID_TIME_METRIC):
                    summary_values.append(tf.summary.Summary.Value(
                        tag=name,
                        simple_value=value
                    ))
            if summary_values:
                summary_obj = tf.summary.Summary(value=summary_values)
                self._summary_writer.add_summary(
                    summary=summary_obj,
                    global_step=self.step_for_metrics,
                )

    def add_summaries(self, summaries, add_metrics=True):
        """Add one-step TensorFlow summaries.
        
        This method will extract metric values from these summaries,
        and call `add_metrics` with these metrics.
        
        Parameters
        ----------
        summaries : collections.Iterable[bytes]
            Computed summary objects.
            
        add_metrics : bool
            Whether or not to add scalars from `summaries` to metrics?
            Default is True.
        """
        # parse the summary objects
        parsed = []
        summaries = tuple(summaries)
        if not summaries:
            return

        for summary in summaries:
            if isinstance(summary, bytes):
                parsed.append(tf.summary.Summary.FromString(summary))
            elif isinstance(summary, tf.summary.Summary):
                parsed.append(summary)
            else:
                raise TypeError('%r is not a TensorFlow summary.' % (summary,))

        # extract metrics from the summary objects
        if add_metrics:
            metrics = {}
            for summary in parsed:
                for value in summary.value:
                    scalar = value.simple_value
                    if isinstance(scalar, (float,) + six.integer_types):
                        metrics[value.tag] = scalar
            if metrics:
                self._add_metrics(six.iteritems(metrics))

        # write the summaries to disk by `tf.summary.FileWriter`
        if self._summary_writer:
            for summary in summaries:
                self._summary_writer.add_summary(
                    summary,
                    global_step=self.step_for_metrics
                )

    def clear_metrics(self):
        """Clear the accumulated metrics."""
        self._metrics.clear()

    def _commit_epoch_time(self):
        if self._epoch_start_time is not None:
            self.add_metrics({
                EPOCH_TIME_METRIC: time.time() - self._epoch_start_time
            })
            self._epoch_start_time = None

    def _commit_step_time(self):
        if self._step_start_time is not None:
            self.add_metrics({
                STEP_TIME_METRIC: time.time() - self._step_start_time
            })
            self._step_start_time = None

    @contextmanager
    def enter_epoch(self):
        """Enter a training epoch context.
        
        Yields
        ------
        int
            The epoch counter.
        """
        try:
            self._within_epoch = True
            self._epoch_start_time = time.time()
            yield self.epoch
            self._commit_epoch_time()
        finally:
            self.epoch += 1
            self._epoch_start_time = None
            self._within_epoch = False

            # flush the summary writer after each epoch
            if self._summary_writer:
                self._summary_writer.flush()

    @contextmanager
    def enter_step(self):
        """Enter a training step context.
        
        Returns
        -------
        int
            The step counter.
        """
        try:
            self._within_step = True
            self._step_start_time = time.time()
            yield self.step
            self._commit_step_time()
        finally:
            self.step += 1
            self._step_start_time = None
            self._within_step = False

    @contextmanager
    def enter_valid(self):
        """Enter a validation context."""
        start_time = time.time()
        yield
        self.add_metrics({
            VALID_TIME_METRIC: time.time() - start_time
        })

    def _format_metric_value(self, val):
        return self.metric_format % val

    def _sorted_metrics(self):
        def push_metric(key, metric, formatter=None):
            val = metric.value
            ret[key] = (
                metric,
                (formatter or self._format_metric_value)(val)
            )

        ret = OrderedDict()
        for key, formatter in [(EPOCH_TIME_METRIC, _humanize_time),
                               (STEP_TIME_METRIC, _humanize_time),
                               (VALID_TIME_METRIC, _humanize_time),
                               (LOSS_METRIC, None),
                               (TRAIN_LOSS_METRIC, None),
                               (VALID_LOSS_METRIC, None)]:
            if key in self._metrics:
                push_metric(key, self._metrics[key], formatter)
        for key in sorted(six.iterkeys(self._metrics)):
            if key not in ret:
                push_metric(key, self._metrics[key])
        return ret

    def _get_messages(self, time_metric, name, counter, max_counter):
        ret = []
        cap_name = name.capitalize()
        if max_counter is not None:
            ret.append('%s %d/%d' % (cap_name, counter + 1, max_counter))
        else:
            ret.append('%s %d' % (cap_name, counter + 1,))

        metrics = self._sorted_metrics()
        if time_metric in metrics:
            timer, formatted = metrics.pop(time_metric)
            if timer.counter > 1:
                fmt = ': %ss finished in avg %%s' % name
            else:
                fmt = ': %s finished in %%s' % name
            ret[-1] += fmt % formatted

        compact_mode = self.compact_log or len(metrics) < 2
        if metrics:
            def metric_name(m, k):
                k = k.replace('_', ' ')
                if m.counter > 1:
                    return 'avg %s' % (k,)
                return k

            if compact_mode:
                ret.extend((
                    '%s: %s' % (metric_name(m, k), s)
                    for k, (m, s) in six.iteritems(metrics)
                ))
            else:
                metric_indent = max(len(k) for k in six.iterkeys(metrics))
                metric_indent = (
                    ((metric_indent + 2) + self.metric_name_width - 1) //
                    self.metric_name_width * self.metric_name_width
                )
                metric_fmt = '%' + str(metric_indent) + 's: %s'
                ret.extend((
                    metric_fmt % (metric_name(m, k), s)
                    for k, (m, s) in six.iteritems(metrics)
                ))
        return ('; ' if compact_mode else '\n').join(ret)

    def get_epoch_log(self):
        """Get epoch(s) log messages.
        
        This method will commit and clear the timer of current epoch.
        """
        self._commit_epoch_time()
        return self._get_messages(
            EPOCH_TIME_METRIC, 'epoch', self.epoch_for_metrics, self.max_epoch)

    def print_epoch_log(self):
        """Print epoch(s) log messages.
        
        This method will commit and clear the timer of current epoch.
        It will also clear all the metrics afterwards.
        """
        self.print_function(self.get_epoch_log())
        self.clear_metrics()

    def get_step_log(self):
        """Get step(s) log messages.
        
        This method will commit and clear the timer of current step.
        """
        self._commit_step_time()
        return self._get_messages(
            STEP_TIME_METRIC, 'step', self.step_for_metrics, self.max_step)

    def print_step_log(self):
        """Print step(s) log messages.
        
        This method will commit and clear the timer of current step.
        It will also clear all the metrics afterwards.
        """
        self.print_function(self.get_step_log())
        self.clear_metrics()
