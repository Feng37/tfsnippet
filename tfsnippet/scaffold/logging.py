# -*- coding: utf-8 -*-
import sys
import re
import time
from collections import defaultdict, OrderedDict
from contextlib import contextmanager

import six
import numpy as np
import tensorflow as tf
import pandas as pd

from tfsnippet.utils import (get_default_session_or_error,
                             is_integer,
                             is_float,
                             VarScopeObject)

__all__ = [
    'TrainLogger',
    'get_parameters_summary',
    'print_parameters_summary',
]

_NOT_SET = object()
_EPOCH_TIME = 'epoch_time'
_STEP_TIME = 'step_time'
_VALID_TIME = 'valid_time'
_LOSS_METRICS = re.compile(r'(^|.*[_/])loss$')
_ACC_METRICS = re.compile(r'(^|.*[_/])acc(uracy)?$')
_TIME_METRICS = re.compile(r'(^|.*[_/])timer?$')
_EPOCH_RELATED_METRICS = [_EPOCH_TIME]


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


class _MetricBestTracker(object):
    """Object to track the best values for certain metrics."""

    def __init__(self, smaller_is_better=True, eps=np.finfo(float).eps):
        self.smaller_is_better = smaller_is_better
        self.eps = eps
        self.value = None

    @property
    def has_value(self):
        return self.value is not None

    def get_value(self):
        return self.value

    def reset(self):
        self.value = None

    def submit(self, metric):
        metric = float(metric)
        if self.value is None:
            self.value = metric
        elif self.smaller_is_better:
            self.value = min(self.value, metric)
        else:
            self.value = max(self.value, metric)

    def is_best(self, metric):
        if self.value is None:
            return True
        elif self.smaller_is_better:
            return metric <= self.value + self.eps
        else:
            return metric >= self.value - self.eps


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


def _default_metric_formatter(name, value):
    if _TIME_METRICS.match(name):
        return _humanize_time(value)
    else:
        return '%.6g' % value


def _get_tensor_value(v):
    if not (is_integer(v) or is_float(v) or isinstance(v, np.ndarray)):
        with tf.name_scope('get_tensor_value', values=[v]):
            v = get_default_session_or_error().run(tf.convert_to_tensor(v))
    return v


class TrainLogger(VarScopeObject):
    """Loss and variables logger.

    This class provides convenient methods for printing training metrics,
    and for writing metrics summaries onto disk.  For example:

        logger = TrainLogger()
        with logger.enter_epoch():
            for batch in minibatch_iterator(data, ignore_incomplete_batch=True):
                with logger.enter_step():
                    loss, _ = session.run([loss, train_op], ...)
                    logger.add_metrics(loss=loss)
            with logger.enter_valid():
                valid_loss = session.run([loss], ...)
                logger.add_metrics(valid_loss=valid_loss)
            logger.print_epoch_log()

    Parameters
    ----------
    initial_epoch : int | tf.Tensor | tf.Variable
        Specify the initial epoch of the training process. 
        It should be one less than the first logged epoch (default 0).
        
    initial_step : int | tf.Tensor | tf.Variable
        Specify the initial step of the training process. 
        It should be one less than the first logged step (default 0).
        
    max_epoch : int
        Specify the maximum epoch of the training process. (default None)
        
    max_step : int
        Specify the maximum step of the training process. (default None)

    metric_formatter : (str, any) -> str
        String formatter for metrics, accept two arguments (name, value),
        and return the formatted metric as string.
        
        (default formatter should format `time`, `timer` or the metrics
         with `_time` or `_timer` as name suffix into human readable 
         time duration, and other metrics as numbers of up to 6 effective
         digits, i.e., formatting via '%.6g').

    print_function : (message) -> None
        Function to print the training logs.

    summary_writer : tf.summary.FileWriter
        TensorFlow summary writer, if specified, will write metric summaries.
        
    summary_exclude_metrics : collections.Iterable[str | re.__Regex]
        Specify the metric patterns excluded from TensorFlow summary.
        (default: `time`, `timer`, or the metrics with `_time` or 
         `_timer` as name suffix)
        
    smaller_better_metrics : collections.Iterable[str | re.__Regex]
        Specify the metric patterns whose smallest values should be tracked.
        (default: `loss`, or the metrics with `_loss` as name suffix)
    
    larger_better_metrics : collections.Iterable[str | re.__Regex]
        Specify the metric patterns whose largest values should be tracked.
        (default: `accuracy`, `acc`, or the metrics with `_acc` or 
         `_accuracy` as name suffix)
        
        If a metric is also specified in `smaller_better_metrics`, it will
        be ignored by `larger_better_metrics`. 
    
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
                 metric_formatter=_default_metric_formatter,
                 print_function=_print_function,
                 summary_writer=None,
                 summary_exclude_metrics=(_TIME_METRICS,),
                 smaller_better_metrics=(_LOSS_METRICS,),
                 larger_better_metrics=(_ACC_METRICS,),
                 name=None,
                 default_name=None):
        # check the arguments
        initial_epoch = int(_get_tensor_value(initial_epoch))
        initial_step = int(_get_tensor_value(initial_step))
        if max_epoch is not None:
            max_epoch = int(max_epoch)
        if max_step is not None:
            max_step = int(max_step)

        # initialize the attributes
        super(TrainLogger, self).__init__(name, default_name)
        self.initial_epoch = initial_epoch
        self.initial_step = initial_step
        self.epoch = initial_epoch  # type: int
        self.step = initial_step    # type: int
        self.max_epoch = max_epoch
        self.max_step = max_step
        self._metric_formatter = metric_formatter
        self._print_function = print_function
        self._summary_writer = summary_writer
        self._summary_exclude_metrics = tuple(summary_exclude_metrics or ())
        self._smaller_better_metrics = tuple(smaller_better_metrics or ())
        self._larger_better_metrics = tuple(larger_better_metrics or ())

        # accumulators for various metrics
        self._metrics = defaultdict(_MetricAccumulator)

        # best value trackers
        self._best_tracker = {}

        # dict to cache the result of summary excluded metric test
        self._summary_exclude_test = {}

        # timer for epoch and step
        self._epoch_start_time = None
        self._step_start_time = None
        self._within_epoch = False
        self._within_step = False

    def reset(self):
        """Reset all internal states.
        
        Note that `summary_writer` is not managed by this object, thus
        cannot be reset.  Reset a `TrainLogger` instance with an external
        `summary_writer` should make no sense.
        """
        self.epoch = self.initial_epoch
        self.step = self.initial_step
        self._metrics.clear()
        self._best_tracker.clear()
        self._summary_exclude_test.clear()
        self._epoch_start_time = self._step_start_time = None
        self._within_epoch = self._within_step = False

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

    def is_best_metric(self, name, value):
        """Check whether or not `value` for metric `name` is the best value.
        
        Parameters
        ----------
        name : str
            Name of the metric.
            
        value : value
            Value of the metric.
        
        Returns
        -------
        bool | None
            True or False if `name` is configured to track the best value.
            None if `name` is not configured.
        """
        best_tracker = self._best_tracker.get(name, None)
        if best_tracker is not None:
            return best_tracker.is_best(value)

    @staticmethod
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

    def _is_summary_excluded_metric(self, name):
        ret = self._summary_exclude_test.get(name, None)
        if ret is None:
            ret = self._test_patterns(self._summary_exclude_metrics, name)
            self._summary_exclude_test[name] = ret
        return ret

    def _get_best_tracker(self, name):
        ret = self._best_tracker.get(name, _NOT_SET)
        if ret is _NOT_SET:
            if self._test_patterns(self._smaller_better_metrics, name):
                ret = _MetricBestTracker(smaller_is_better=True)
            elif self._test_patterns(self._larger_better_metrics, name):
                ret = _MetricBestTracker(smaller_is_better=False)
            else:
                ret = None
            self._best_tracker[name] = ret
        return ret

    def _add_metrics(self, metrics):
        for k, v in metrics:
            self._metrics[k].submit(v)
            best_tracker = self._get_best_tracker(k)
            if best_tracker is not None:
                best_tracker.submit(v)

    def _create_summary_values(self, metrics):
        return [
            tf.summary.Summary.Value(
                tag=name,
                simple_value=value
            )
            for name, value in metrics
            if not self._is_summary_excluded_metric(name)
        ]

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
                    global_step=self.step,
                )

    def _clear_metrics(self, step_only=False):
        """Clear the accumulated metrics."""
        for k, v in six.iteritems(self._metrics):
            if not step_only or k not in _EPOCH_RELATED_METRICS:
                v.reset()

    def _commit_epoch_time(self):
        if self._epoch_start_time is not None:
            self.add_metrics({
                _EPOCH_TIME: time.time() - self._epoch_start_time
            })
            self._epoch_start_time = None

    def _commit_step_time(self):
        if self._step_start_time is not None:
            self.add_metrics({
                _STEP_TIME: time.time() - self._step_start_time
            })
            self._step_start_time = None

    @contextmanager
    def enter_epoch(self):
        """Enter a training epoch context.
        
        Yields
        ------
        int
            The epoch counter (starting from 1).
        """
        try:
            self._within_epoch = True
            self._epoch_start_time = time.time()
            self.epoch += 1
            yield self.epoch
            self._commit_epoch_time()
        finally:
            self._epoch_start_time = None
            self._within_epoch = False

    def iter_epochs(self):
        """Iterate through epochs.
        
        This method will yield the epoch counters.  If `max_epoch` is
        configured, it will stop at `max_epoch` (exclusive).
        Note that this method is not necessary, where the following
        two loops are equivalent:

            # Suppose we have a train logger with `max_epoch` set to 3. 
            # Without it the logger cannot show the max epoch information
            # in log messages
            logger = TrainLogger(max_epoch=3)

            # The first approach: to iterate through epochs via logger.
            for epoch in logger.iter_epochs():
                ...

            # The second approach: to iterate through epochs by hand.
            for epoch in range(1, 4):
                with logger.enter_epoch():
                    ...

        The latter loop will do the exact thing as the first loop.
        Note that the context ``with logger.enter_epoch()`` is necessary
        for the logger to record training time for an epoch.

        Yields
        ------
        int
            The epoch counter (starting from 1).
        """
        while True:
            with self.enter_epoch() as epoch:
                yield epoch
            if self.max_epoch is not None and self.epoch >= self.max_epoch:
                break

    @contextmanager
    def enter_step(self):
        """Enter a training step context.
        
        Returns
        -------
        int
            The step counter (starting from 1).
        """
        try:
            self._within_step = True
            self._step_start_time = time.time()
            self.step += 1
            yield self.step
            self._commit_step_time()
        finally:
            self._step_start_time = None
            self._within_step = False

    def iter_steps(self, data_iterator):
        """Iterate through steps and epoch data batches.
        
        This method will yield the step counters, as well as the data
        batches in an epoch.  It will stop at `max_step`, or when
        `data_iterator` has been exhausted.
        
        Note that this method is not necessary, just using `enter_step`
        will be enough.  See `iter_epochs` for more details.
        
        Parameters
        ----------
        data_iterator
            The batch data iterator.
            
        Yields
        ------
        (int, any)
            The step counter (starting from 1) as well as the batch data.
        """
        for batch in data_iterator:
            with self.enter_step() as step:
                yield step, batch
            if self.max_step is not None and self.step >= self.max_step:
                break

    @contextmanager
    def enter_valid(self):
        """Enter a validation context."""
        start_time = time.time()
        yield
        self.add_metrics({
            _VALID_TIME: time.time() - start_time
        })

    def _sorted_metrics(self):
        def sort_key(name):
            if _TIME_METRICS.match(name):
                return -3, name
            elif _LOSS_METRICS.match(name):
                return -2, name
            elif _ACC_METRICS.match(name):
                return -1, name
            return 0, name

        ret = OrderedDict()
        for key in sorted(six.iterkeys(self._metrics), key=sort_key):
            if key not in ret:
                metric = self._metrics[key]
                if metric.has_value:
                    ret[key] = (
                        metric,
                        self._metric_formatter(key, metric.value),
                        self.is_best_metric(key, metric.value)
                    )
        return ret

    def _get_messages(self, time_metric, name, counter_tag, step_only=False):

        def best_mark(b):
            return ' (*)' if b else ''

        ret = [counter_tag]
        metrics = self._sorted_metrics()
        if time_metric in metrics:
            timer, formatted, is_best = metrics.pop(time_metric)
            if timer.counter > 1:
                fmt = ': %%s/%s' % name
            else:
                fmt = ': finished in %s'
            ret[-1] += fmt % formatted

        if metrics:
            def metric_name(m, k):
                pos = k.rfind('/')
                if pos >= 0:
                    k = k[pos+1:]
                k = k.replace('_', ' ')
                if m.counter > 1:
                    return 'avg %s' % (k,)
                return k

            ret.extend((
                '%s: %s%s' % (metric_name(m, k), s, best_mark(b))
                for k, (m, s, b) in six.iteritems(metrics)
                if not step_only or k not in _EPOCH_RELATED_METRICS
            ))
        return '; '.join(ret)

    @property
    def _epoch_counter_tag(self):
        if self.max_epoch is None:
            return 'epoch %d' % (self.epoch,)
        else:
            return 'epoch %d/%d' % (self.epoch, self.max_epoch)

    def get_epoch_log(self):
        """Get epoch(s) log messages.
        
        This method will commit and clear the timer of current epoch.
        It will also clear all the metrics afterwards.
        """
        self._commit_epoch_time()
        counter_tag = self._epoch_counter_tag.capitalize()
        ret = self._get_messages(_EPOCH_TIME, 'epoch', counter_tag)
        self._clear_metrics()
        return ret

    def print_epoch_log(self):
        """Print epoch(s) log messages.
        
        This method will commit and clear the timer of current epoch.
        It will also clear all the metrics afterwards.
        """
        self._print_function(self.get_epoch_log())

    @property
    def _step_counter_tag(self):
        if self.max_step is None:
            return 'step %d' % (self.step,)
        else:
            return 'step %d/%d' % (self.step, self.max_step)

    def get_step_log(self):
        """Get step(s) log messages.
        
        This method will commit and clear the timer of current step.
        It will also clear all the metrics afterwards.
        """
        self._commit_step_time()
        counter_tag = '%s, %s' % (
            self._epoch_counter_tag.capitalize(), self._step_counter_tag)
        ret = self._get_messages(_STEP_TIME, 'step', counter_tag,
                                 step_only=True)
        self._clear_metrics(step_only=True)
        return ret

    def print_step_log(self):
        """Print step(s) log messages.
        
        This method will commit and clear the timer of current step.
        It will also clear all the metrics afterwards.
        """
        self._print_function(self.get_step_log())


def get_parameters_summary(variables):
    """Get a formatted summary about the parameters.
    
    Parameters
    ----------
    variables : list[tf.Variable] | dict[str, tf.Variable]
        List or dict of variables, which should be optimized during training.
        
    Returns
    -------
    list[str]
        Formatted message lines (including line endings) about the training.
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
    var_count = [np.prod(s.as_list()) for s in var_shape]
    var_shape = [str(s) for s in var_shape]
    var_table = pd.DataFrame(
        data=OrderedDict([
            ('a', var_name),
            ('b', var_shape),
            ('c', np.asarray(var_count, dtype=np.int32))
        ])
    )

    if len(var_count) > 0:
        var_table_str = '\n'.join([
            s.rstrip() for s in var_table.to_string(index=False).split('\n')
        ][1:])
        max_line_length = max(
            len(s.rstrip()) for s in var_table_str.split('\n'))
        buf.append('Trainable Parameters (%d in total)\n' % sum(var_count))
        max_line_length = max(max_line_length, len(buf[-1]) - 1)
        buf[-1] = (' ' * (max_line_length - len(buf[-1]) + 1) + buf[-1])
        buf.append('-' * max_line_length + '\n')
        buf.append(var_table_str)

    return buf


def print_parameters_summary(variables, print_function=_print_function):
    """Print formatted summary about the parameters.
    
    Parameters
    ----------
    variables : list[tf.Variable] | dict[str, tf.Variable]
        List or dict of variables, which should be optimized during training.

    print_function : (message) -> None
        Function to print the training logs.
    """
    summary = get_parameters_summary(variables)
    print_function(''.join(summary))
