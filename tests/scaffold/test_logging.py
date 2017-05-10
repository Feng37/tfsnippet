# -*- coding: utf-8 -*-
import contextlib
import os
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.scaffold import (get_variables_summary, MetricLogger,
                                SummaryWriter)
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

            self.assertEqual(get_variables_summary([]), '')
            self.assertEqual(get_variables_summary([a]), (
                'Variables Summary (2 in total)\n'
                '------------------------------\n'
                'a  (2,)  2'
            ))
            self.assertEqual(get_variables_summary([a, b, c]), (
                'Variables Summary (63 in total)\n'
                '-------------------------------\n'
                'a         (2,)       2\n'
                'c         ()         1\n'
                'nested/b  (3, 4, 5)  60'
            ))
            self.assertEqual(get_variables_summary({'a': a, 'b': b, 'c': c}), (
                'Variables Summary (63 in total)\n'
                '-------------------------------\n'
                'a  (2,)       2\n'
                'b  (3, 4, 5)  60\n'
                'c  ()         1'
            ))


class MetricLoggerTestCase(TestCase):

    def test_basic_logging(self):
        logger = MetricLogger()
        self.assertEqual(logger.format_logs(), '')

        logger.add_metrics(loss=1.)
        logger.add_metrics(loss=2., valid_loss=3., valid_timer=0.1)
        logger.add_metrics(loss=4., valid_acc=5., train_time=0.2)
        logger.add_metrics(loss=6., valid_acc=7., train_time=0.3)
        self.assertEqual(
            logger.format_logs(),
            'train time: 0.25 sec (±0.05 sec); '
            'valid timer: 0.1 sec; '
            'loss: 3.25 (±1.92029); '
            'valid loss: 3; '
            'valid acc: 6 (±1)'
        )

        logger.clear()
        self.assertEqual(logger.format_logs(), '')

        with tf.Graph().as_default(), tf.Session().as_default():
            logger.add_metrics(metrics={'loss': 1.})
            self.assertEqual(logger.format_logs(), 'loss: 1')

    def test_errors(self):
        logger = MetricLogger()
        with self.assertRaisesRegex(TypeError, '`metrics` should be a dict.'):
            logger.add_metrics(metrics=[])


class SummaryWriterTestCase(TestCase):

    def test_basic(self):
        with TemporaryDirectory() as tempdir:
            # generate the metric summary
            with contextlib.closing(SummaryWriter(
                    tf.summary.FileWriter(tempdir))) as sw:
                step = 0
                for epoch in range(1, 3):
                    for data in range(10):
                        step += 1
                        sw.add_metrics(global_step=step, acc=step * 100 + data)

                    with tf.Graph().as_default(), tf.Session().as_default():
                        sw.add_metrics(
                            global_step=tf.constant(step),
                            metrics={'valid_loss': -epoch}
                        )

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
