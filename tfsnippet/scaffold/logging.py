# -*- coding: utf-8 -*-
import re
from collections import defaultdict, OrderedDict

import six
import numpy as np
import tensorflow as tf

from tfsnippet.utils import (humanize_duration, MetricAccumulator,
                             get_default_session_or_error)

__all__ = [
    'MetricFormatter',
    'MetricLogger',
    'SummaryWriter',
    'get_parameters_summary',
]

_LOSS_METRICS = re.compile(r'(^|.*[_/])loss$')
_ACC_METRICS = re.compile(r'(^|.*[_/])acc(uracy)?$')
_TIME_METRICS = re.compile(r'(^|.*[_/])timer?$')


class MetricFormatter(object):
    """Default class for training metric formatter.

    A training metric formatter should take a set of metrics, placing these
    metrics in pre-defined order, than format these metrics values into texts.
    """

    def sort_metrics(self, metrics):
        """Sort the specified `metrics`.

        By default this method places metrics ending with "time" or "timer"
        at first, metrics ending with "loss" at second, metrics ending with
        "acc" or "accuracy" at third, and other remaining metrics the last.
        Metrics within the same group will be sorted according to their name.

        Parameters
        ----------
        metrics : dict[str, any]
            The metrics to be sorted.

        Returns
        -------
        list[(str, any)]
            List of (key, value) of these metrics.
        """
        def sort_key(name):
            if _TIME_METRICS.match(name):
                return -3, name
            elif _LOSS_METRICS.match(name):
                return -2, name
            elif _ACC_METRICS.match(name):
                return -1, name
            return 0, name

        return [
            (key, metrics[key])
            for key in sorted(six.iterkeys(metrics), key=sort_key)
        ]

    def format_metric_value(self, name, value):
        """Format the value of specified metric.

        Parameters
        ----------
        name : str
            Name of the metric.

        value : any
            Value of the metric.

        Returns
        -------
        str
            Human readable string representation of the metric value.
        """
        if _TIME_METRICS.match(name):
            return humanize_duration(value)
        else:
            return '%.6g' % (value,)


class MetricLogger(object):
    """Training metrics logger.

    This class provides convenient methods for logging training metrics,
    and for writing metrics onto disk by TensorFlow summary writer.

    An example of using this logger is:

        logger = MetricLogger()
        step = 0
        for epoch in range(1, max_epoch+1):
            for batch in minibatch_iterator(data, ignore_incomplete_batch=True):
                step += 1
                loss, _ = session.run([loss, train_op], ...)
                logger.add_metrics(step, loss=loss)

            valid_loss = session.run([loss], ...)
            logger.add_metrics(valid_loss=valid_loss)
            print('Epoch %d: %s' %  (epoch, step, logger.format_logs()))
            logger.clear()

    Parameters
    ----------
    formatter : MetricFormatter
        Optional metric formatter for this logger.
    """

    def __init__(self, formatter=None):
        if formatter is None:
            formatter = MetricFormatter()
        self._formatter = formatter

        # accumulators for various metrics
        self._metrics = defaultdict(MetricAccumulator)

    def clear(self):
        """Clear all the metrics."""
        # instead of calling ``self._metrics.clear()``, we reset every
        # accumulator object (so that they can be reused)
        for k, v in six.iteritems(self._metrics):
            v.reset()

    def add_metrics(self, metrics=None, **kwargs):
        """Add one-step metric values.

        This method will compose TensorFlow summary objects from
        these metrics, and write to disk with these summaries.

        Parameters
        ----------
        metrics, **kwargs
            One-step metric values.
        """
        if metrics is not None and not isinstance(metrics, dict):
            raise TypeError('`metrics` should be a dict.')
        if metrics:
            for k, v in six.iteritems(metrics):
                self._metrics[k].add(v)
        if kwargs:
            for k, v in six.iteritems(kwargs):
                self._metrics[k].add(v)

    def format_logs(self):
        """Format the metrics logs.

        Returns
        -------
        str
            The formatted log messages text.
        """
        buf = []
        for k, m in self._formatter.sort_metrics(self._metrics):
            if m.has_value:
                pfx = 'avg ' if m.counter > 1 else ''
                val = self._formatter.format_metric_value(k, m.value)
                name = k.replace('_', ' ')
                buf.append('%s%s: %s' % (pfx, name, val))
        return '; '.join(buf)


class SummaryWriter(object):
    """Wrapper for TensorFlow summary writer.

    This class wraps the TensorFlow summary writer, providing auxiliary
    methods for convenience.

    Parameters
    ----------
    writer : tf.summary.FileWriter
        The TensorFlow summary writer instance.
    """

    def __init__(self, writer):
        self._writer = writer

    def close(self):
        """Close the underlying summary writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def add_metrics(self, global_step=None, metrics=None, **kwargs):
        """Add a scalar metric as summary.

        Parameters
        ----------
        global_step : int | tf.Tensor | tf.Variable
            The global step counter. (optional)

        metrics, **kwargs
            Dict of metric values.
        """
        if metrics is not None and not isinstance(metrics, (dict, OrderedDict)):
            raise TypeError('%r should be a dict.' % (metrics,))

        values = []
        if metrics:
            for k, v in six.iteritems(metrics):
                values.append(tf.summary.Summary.Value(tag=k, simple_value=v))
        for k, v in six.iteritems(kwargs):
            values.append(tf.summary.Summary.Value(tag=k, simple_value=v))

        if values:
            if isinstance(global_step, (tf.Tensor, tf.Variable)):
                global_step = get_default_session_or_error().run(global_step)
            summary = tf.summary.Summary(value=values)
            self._writer.add_summary(summary, global_step=global_step)

    def add_summary(self, summary, global_step=None):
        """Add a summary object.

        Parameters
        ----------
        summary : bytes | tf.summary.Summary
            The summary object.

        global_step : int | tf.Tensor | tf.Variable
            The global step counter. (optional)
        """
        if isinstance(global_step, (tf.Tensor, tf.Variable)):
            global_step = get_default_session_or_error().run(global_step)
        self._writer.add_summary(summary, global_step=global_step)


def get_parameters_summary(variables):
    """Get a formatted summary about the parameters.

    Parameters
    ----------
    variables : list[tf.Variable] | dict[str, tf.Variable]
        List or dict of variables, which should be optimized during training.

    Returns
    -------
    str
        Formatted message about the training.
    """
    buf = []

    # collect trainable variable summaries
    if isinstance(variables, dict):
        var_name, var_shape = list(zip(*sorted(six.iteritems(variables))))
        var_shape = [s.get_shape() for s in var_shape]
    else:
        variables = sorted(variables, key=lambda v: v.name)
        var_name = [v.name.rsplit(':', 1)[0] for v in variables]
        var_shape = [v.get_shape() for v in variables]
    var_count = [int(np.prod(s.as_list(), dtype=np.int32)) for s in var_shape]
    var_count_total = sum(var_count)
    var_shape = [str(s) for s in var_shape]
    var_count = [str(s) for s in var_count]

    if len(var_count) > 0:
        var_title = 'Trainable Parameters (%d in total)' % (var_count_total,)

        x, y, z = (max(map(len, var_name)), max(map(len, var_shape)),
                   max(map(len, var_count)))
        var_format = '%%-%ds  %%-%ds  %%s' % (x, y)
        var_table = []
        for name, shape, count in zip(var_name, var_shape, var_count):
            var_table.append(var_format % (name, shape, count))

        max_line_length = max(x + y + z + 4, len(var_title))
        buf.append(var_title)
        buf.append('-' * max_line_length)
        buf.extend(var_table)

    return '\n'.join(buf)
