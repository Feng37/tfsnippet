# -*- coding: utf-8 -*-
import itertools
import os
import re
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.scaffold import get_parameters_summary, MetricLogger
from tfsnippet.utils import TemporaryDirectory
from tests.helper import TestCase


class LoggingUtilsTestCase(TestCase):

    def test_get_parameters_summary(self):
        with tf.Graph().as_default():
            # test variable summaries
            a = tf.get_variable('a', dtype=tf.int32, shape=[2])
            with tf.variable_scope('nested'):
                b = tf.get_variable('b', dtype=tf.float32, shape=(3, 4, 5))
            c = tf.get_variable('c', dtype=tf.float32, shape=())

            self.assertEqual(get_parameters_summary([]), '')
            self.assertEqual(get_parameters_summary([a]), (
                'Trainable Parameters (2 in total)\n'
                '---------------------------------\n'
                'a  (2,)  2'
            ))
            self.assertEqual(get_parameters_summary([a, b, c]), (
                'Trainable Parameters (63 in total)\n'
                '----------------------------------\n'
                'a         (2,)       2\n'
                'c         ()         1\n'
                'nested/b  (3, 4, 5)  60'
            ))
            self.assertEqual(get_parameters_summary({'a': a, 'b': b, 'c': c}), (
                'Trainable Parameters (63 in total)\n'
                '----------------------------------\n'
                'a  (2,)       2\n'
                'b  (3, 4, 5)  60\n'
                'c  ()         1'
            ))


class MetricLoggerTestCase(TestCase):

    def test_summary_exclusion(self):
        # default patterns
        logger = MetricLogger()
        for prefix, name in itertools.product(['', 'xxx_', '/', 'xxx/yyy_'],
                                              ['time', 'timer', 'loss',
                                               'acc', 'accuracy']):
            # test for double times, to check the internal cache for patterns
            self.assertFalse(logger._is_summary_excluded_metric(prefix + name))
            self.assertFalse(logger._is_summary_excluded_metric(prefix + name))

        # customized patterns
        logger = MetricLogger(
            summary_excluded_metrics=[
                'loss', 'xxx_loss', re.compile(r'^.*/.*loss$')])
        for prefix, name in itertools.product(['', 'xxx_', '/', 'xxx/yyy_'],
                                              ['loss']):
            self.assertTrue(logger._is_summary_excluded_metric(prefix + name))
            self.assertTrue(logger._is_summary_excluded_metric(prefix + name))
        for prefix, name in itertools.product(['', 'xxx_', '/', 'xxx/yyy_'],
                                              ['acc', 'accuracy', 'time',
                                               'timer']):
            self.assertFalse(logger._is_summary_excluded_metric(prefix + name))
            self.assertFalse(logger._is_summary_excluded_metric(prefix + name))

    def test_basic_logging(self):
        logger = MetricLogger()
        self.assertEqual(logger.format_logs(), '')

        logger.add_metrics(1, loss=1.)
        logger.add_metrics(2, loss=2., valid_loss=3., valid_timer=0.1)
        logger.add_metrics(3, loss=4., valid_acc=5., train_time=0.2)
        logger.add_metrics(4, loss=6., valid_acc=7., train_time=0.3)
        self.assertEqual(
            logger.format_logs(),
            'avg train time: 0.25 sec; valid timer: 0.1 sec; avg loss: 3.25; '
            'valid loss: 3; avg valid acc: 6'
        )

        logger.clear()
        self.assertEqual(logger.format_logs(), '')

        with tf.Graph().as_default(), tf.Session().as_default():
            logger.add_metrics(tf.constant(1), metrics={'loss': 1.})
            self.assertEqual(logger.format_logs(), 'loss: 1')

    def test_errors(self):
        logger = MetricLogger()
        with self.assertRaisesRegex(TypeError, '`metrics` should be a dict.'):
            logger.add_metrics(metrics=[])

    def test_tensorflow_summary(self):
        with TemporaryDirectory() as tempdir:
            # generate the metric summary
            sw = tf.summary.FileWriter(tempdir)
            logger = MetricLogger(summary_writer=sw)
            step = 0
            for epoch in range(1, 3):
                for data in range(10):
                    step += 1
                    logger.add_metrics(global_step=step, acc=step * 100 + data)
                logger.add_metrics(
                    global_step=step,
                    metrics={'valid_loss': -epoch}
                )
            sw.close()

            # read the metric summary
            acc_steps = []
            acc_values = []
            valid_loss_steps = []
            valid_loss_values = []
            tags = set()

            event_file_path = os.path.join(tempdir, os.listdir(tempdir)[0])
            for e in tf.train.summary_iterator(event_file_path):
                for v in e.summary.value:
                    tags.add(v.tag)
                    if v.tag == 'acc':
                        acc_steps.append(e.step)
                        acc_values.append(v.simple_value)
                    elif v.tag == 'valid_loss':
                        valid_loss_steps.append(e.step)
                        valid_loss_values.append(v.simple_value)

            self.assertEqual(sorted(tags), ['acc', 'valid_loss'])
            np.testing.assert_equal(acc_steps, np.arange(1, 21))
            np.testing.assert_almost_equal(
                acc_values,
                np.arange(1, 21) * 100 + np.concatenate([
                    np.arange(10), np.arange(10)
                ])
            )
            np.testing.assert_equal(valid_loss_steps, [10, 20])
            np.testing.assert_almost_equal(
                valid_loss_values,
                [-1, -2]
            )


if __name__ == '__main__':
    unittest.main()
