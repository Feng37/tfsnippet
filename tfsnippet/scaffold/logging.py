# -*- coding: utf-8 -*-
import re
from collections import defaultdict

import six
import numpy as np
import tensorflow as tf

from tfsnippet.utils import (humanize_duration, MetricAccumulator,
                             get_default_session_or_error)

__all__ = [
    'MetricFormatter',
    'MetricLogger',
    'get_parameters_summary',
]

_LOSS_METRICS = re.compile(r'(^|.*[_/])loss$')
_ACC_METRICS = re.compile(r'(^|.*[_/])acc(uracy)?$')
_TIME_METRICS = re.compile(r'(^|.*[_/])timer?$')


def _test_patterns(patterns, value):
    if patterns:
        for pattern in patterns:
            if isinstance(pattern, six.string_types):
                if pattern == value:
                    return True
            else:
                if pattern.match(value):
                    return True
    return False


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
    
    summary_writer : tf.summary.FileWriter
        Optional TensorFlow summary writer for the metrics.

    summary_excluded_metrics : collections.Iterable[str | re.__Regex]
        Specify the metric patterns excluded from TensorFlow summary.
    """

    def __init__(self,
                 formatter=None,
                 summary_writer=None,
                 summary_excluded_metrics=()):
        if formatter is None:
            formatter = MetricFormatter()
        self._formatter = formatter
        self._summary_writer = summary_writer
        self._summary_excluded_metrics = summary_excluded_metrics

        # cache to store the summary names already tested against
        # `_summary_excluded_metrics`
        self._summary_excluded_metrics_cache = {}

        # accumulators for various metrics
        self._metrics = defaultdict(MetricAccumulator)

    def clear(self):
        """Clear all the metrics."""
        # instead of calling ``self._metrics.clear()``, we reset every
        # accumulator object (so that they can be reused)
        for k, v in six.iteritems(self._metrics):
            v.reset()

    def add_metrics(self, global_step=None, metrics=None, **kwargs):
        """Add one-step metric values.

        This method will compose TensorFlow summary objects from
        these metrics, and write to disk with these summaries.

        Parameters
        ----------
        global_step : int | tf.Variable | tf.Tensor
            The global step counter.
        
        metrics, **kwargs
            One-step metric values.
        """
        if metrics is not None and not isinstance(metrics, dict):
            raise TypeError('`metrics` should be a dict.')
        if isinstance(global_step, (tf.Tensor, tf.Variable)):
            global_step = get_default_session_or_error().run(global_step)

        if metrics:
            self._add_metrics(six.iteritems(metrics))
        if kwargs:
            self._add_metrics(six.iteritems(kwargs))

        if self._summary_writer:
            summary_values = []
            if metrics:
                summary_values.extend(
                    self._create_summary_values(six.iteritems(metrics)))
            if kwargs:
                summary_values.extend(
                    self._create_summary_values(six.iteritems(kwargs)))

            if summary_values:
                summary_obj = tf.summary.Summary(value=summary_values)
                self._summary_writer.add_summary(
                    summary=summary_obj,
                    global_step=global_step,
                )

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

    def _add_metrics(self, metrics):
        for k, v in metrics:
            self._metrics[k].add(v)

    def _is_summary_excluded_metric(self, name):
        ret = self._summary_excluded_metrics_cache.get(name, None)
        if ret is None:
            ret = _test_patterns(self._summary_excluded_metrics, name)
            self._summary_excluded_metrics_cache[name] = ret
        return ret

    def _create_summary_values(self, metrics):
        return [
            tf.summary.Summary.Value(
                tag=name,
                simple_value=value
            )
            for name, value in metrics
            if not self._is_summary_excluded_metric(name)
        ]


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
