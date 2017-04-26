# -*- coding: utf-8 -*-
import re
import time
import unittest

import itertools
import numpy as np
import tensorflow as tf
from tensorflow.python.summary.event_accumulator import EventAccumulator

from mlcomp.utils import TemporaryDirectory
from tfsnippet.scaffold import TrainLogger


class LoggingTestCase(unittest.TestCase):

    def _assertMatch(self, pattern, value):
        self.assertTrue(
            re.match(pattern, value) is not None,
            msg='%r does not match %r' % (value, pattern)
        )

    def test_TrainLogger(self):
        # test default patterns of TrainLogger
        logger = TrainLogger()
        for prefix, name in itertools.product(['', 'xxx_', '/', 'xxx/yyy_'],
                                              ['time', 'timer']):
            self.assertTrue(logger._is_summary_excluded_metric(prefix + name))
            self.assertTrue(logger._is_summary_excluded_metric(prefix + name))
            self.assertIsNone(logger._get_best_tracker(prefix + name))
            self.assertIsNone(logger._get_best_tracker(prefix + name))
        for prefix, name in itertools.product(['', 'xxx_', '/', 'xxx/yyy_'],
                                              ['loss']):
            self.assertFalse(logger._is_summary_excluded_metric(prefix + name))
            self.assertFalse(logger._is_summary_excluded_metric(prefix + name))
            self.assertTrue(logger._get_best_tracker(prefix + name).
                            smaller_is_better)
            self.assertTrue(logger._get_best_tracker(prefix + name).
                            smaller_is_better)
        for prefix, name in itertools.product(['', 'xxx_', '/', 'xxx/yyy_'],
                                              ['acc', 'accuracy']):
            self.assertFalse(logger._is_summary_excluded_metric(prefix + name))
            self.assertFalse(logger._is_summary_excluded_metric(prefix + name))
            self.assertFalse(logger._get_best_tracker(prefix + name).
                             smaller_is_better)
            self.assertFalse(logger._get_best_tracker(prefix + name).
                             smaller_is_better)

        # test customized patterns of TrainLogger
        logger = TrainLogger(
            summary_exclude_metrics=['loss',
                                     'xxx_loss',
                                     re.compile(r'^.*/.*loss$')],
            smaller_better_metrics=[re.compile(r'^acc(uracy)?$'),
                                    'xxx_acc',
                                    'xxx_accuracy',
                                    re.compile(r'^.*/.*acc(uracy)?$')],
            larger_better_metrics=['time',
                                   'timer',
                                   re.compile(r'(^|.*[_/])timer?$')],
        )
        for prefix, name in itertools.product(['', 'xxx_', '/', 'xxx/yyy_'],
                                              ['time', 'timer']):
            self.assertFalse(logger._is_summary_excluded_metric(prefix + name))
            self.assertFalse(logger._is_summary_excluded_metric(prefix + name))
            self.assertFalse(logger._get_best_tracker(prefix + name).
                             smaller_is_better)
            self.assertFalse(logger._get_best_tracker(prefix + name).
                             smaller_is_better)
        for prefix, name in itertools.product(['', 'xxx_', '/', 'xxx/yyy_'],
                                              ['loss']):
            self.assertTrue(logger._is_summary_excluded_metric(prefix + name))
            self.assertTrue(logger._is_summary_excluded_metric(prefix + name))
            self.assertIsNone(logger._get_best_tracker(prefix + name))
            self.assertIsNone(logger._get_best_tracker(prefix + name))
        for prefix, name in itertools.product(['', 'xxx_', '/', 'xxx/yyy_'],
                                              ['acc', 'accuracy']):
            self.assertFalse(logger._is_summary_excluded_metric(prefix + name))
            self.assertFalse(logger._is_summary_excluded_metric(prefix + name))
            self.assertTrue(logger._get_best_tracker(prefix + name).
                            smaller_is_better)
            self.assertTrue(logger._get_best_tracker(prefix + name).
                            smaller_is_better)

        # test step logs of TrainLogger
        logs = []
        logger = TrainLogger(max_epoch=2, max_step=6)
        epoch_counter = 1
        step_counter = 1
        for epoch in logger.iter_epochs():
            self.assertEqual(epoch_counter, epoch)
            epoch_counter += 1
            for step, data in logger.iter_steps([1, 2, 3, 4]):
                self.assertEqual(step_counter, step)
                step_counter += 1
                logger.add_metrics(loss=-data, acc=data)
                if step % 2 == 0:
                    with logger.enter_valid():
                        logger.add_metrics(valid_loss=-step, valid_acc=step)
                    logs.append(logger.get_step_log())

        self._assertMatch(
            re.compile(r'^Step 2/6: avg [^ ]+ sec for every step; '
                       r'valid time: [^ ]+ sec; '
                       r'avg loss: -1\.5; '
                       r'valid loss: -2 \(\*\); '
                       r'avg acc: 1\.5; '
                       r'valid acc: 2 \(\*\)$'),
            logs[0]
        )
        self._assertMatch(
            re.compile(r'^Step 4/6: avg [^ ]+ sec for every step; '
                       r'valid time: [^ ]+ sec; '
                       r'avg loss: -3\.5; '
                       r'valid loss: -4 \(\*\); '
                       r'avg acc: 3\.5; '
                       r'valid acc: 4 \(\*\)$'),
            logs[1]
        )
        self._assertMatch(
            re.compile(r'^Step 6/6: avg [^ ]+ sec for every step; '
                       r'valid time: [^ ]+ sec; '
                       r'avg loss: -1\.5; '
                       r'valid loss: -6 \(\*\); '
                       r'avg acc: 1\.5; '
                       r'valid acc: 6 \(\*\)$'),
            logs[2]
        )

        # test epoch logs of TrainLogger
        logs = []
        # logger = TrainLogger(max_epoch=2, max_step=6)
        logger.reset()      # this is expected to have the same effect as above
        for epoch in logger.iter_epochs():
            s_data = 0.
            s_count = 0
            for step, data in logger.iter_steps([1, 2, 3, 4]):
                logger.add_metrics(loss=-data, acc=data)
                s_data += data
                s_count += 1
                np.testing.assert_almost_equal(
                    s_data / s_count,
                    logger.get_metric('acc')
                )
                self.assertIsNone(logger.get_metric('none_exist_metric'))
            with logger.enter_valid():
                logger.add_metrics(valid_loss=-epoch, valid_acc=epoch)
            logs.append(logger.get_epoch_log())
        self._assertMatch(
            re.compile(r'^Epoch 1/2: finished in [^ ]+ sec; '
                       r'avg step time: [^ ]+ sec; '
                       r'valid time: [^ ]+ sec; '
                       r'avg loss: -2\.5; '
                       r'valid loss: -1 \(\*\); '
                       r'avg acc: 2\.5; '
                       r'valid acc: 1 \(\*\)$'),
            logs[0]
        )
        self._assertMatch(
            re.compile(r'^Epoch 2/2: finished in [^ ]+ sec; '
                       r'avg step time: [^ ]+ sec; '
                       r'valid time: [^ ]+ sec; '
                       r'avg loss: -1\.5; '
                       r'valid loss: -2 \(\*\); '
                       r'avg acc: 1\.5; '
                       r'valid acc: 2 \(\*\)$'),
            logs[1]
        )

        # test logs without `max_epoch` and `max_step`
        logger = TrainLogger()
        with logger.enter_epoch():
            with logger.enter_step():
                logger.add_metrics(loss=1.)
                self._assertMatch(
                    re.compile(r'^Step 1: finished in [^ ]+ sec; '
                               r'loss: 1 \(\*\)$'),
                    logger.get_step_log()
                )
            logger.add_metrics(loss=2.)
            self._assertMatch(
                re.compile(r'^Epoch 1: finished in [^ ]+ sec; '
                           r'loss: 2$'),
                logger.get_epoch_log()
            )

        # test timers
        logs = []
        logger = TrainLogger(max_epoch=2)
        for epoch in logger.iter_epochs():
            for i in range(2):
                with logger.enter_step():
                    time.sleep(0.1 * epoch)
            with logger.enter_valid():
                time.sleep(0.2)
            logs.append(logger.get_epoch_log())
        self._assertMatch(
            re.compile(r'^Epoch 1/2: finished in 0.4[^ ]+ sec; '
                       r'avg step time: 0.1[^ ]+ sec; '
                       r'valid time: 0.2[^ ]+ sec$'),
            logs[0]
        )
        self._assertMatch(
            re.compile(r'^Epoch 2/2: finished in 0.6[^ ]+ sec; '
                       r'avg step time: 0.2[^ ]+ sec; '
                       r'valid time: 0.2[^ ]+ sec$'),
            logs[1]
        )

        # test writing TensorFlow summary with TrainLogger
        with TemporaryDirectory() as tempdir:
            # generate the metric summary
            sw = tf.summary.FileWriter(tempdir)
            logger = TrainLogger(summary_writer=sw, max_epoch=2)
            for epoch in logger.iter_epochs():
                for step, data in logger.iter_steps(range(10)):
                    logger.add_metrics(acc=step * 100 + data)
                logger.add_metrics(valid_loss=-epoch)
            sw.close()

            # read the metric summary
            acc = EventAccumulator(tempdir)
            acc.Reload()
            self.assertEqual(sorted(acc.Tags()['scalars']),
                             ['acc', 'valid_loss'])

            def extract_scalar(tag):
                events = acc.Scalars(tag)
                steps = np.asarray([e.step for e in events], dtype=np.int)
                values = np.asarray([e.value for e in events], dtype=np.float64)
                return steps, values

            acc_steps, acc_values = extract_scalar('acc')
            np.testing.assert_equal(acc_steps, np.arange(1, 21))
            np.testing.assert_almost_equal(
                acc_values,
                np.arange(1, 21) * 100 + np.concatenate([
                    np.arange(10), np.arange(10)
                ])
            )

            valid_loss_steps, valid_loss_values = extract_scalar('valid_loss')
            np.testing.assert_equal(valid_loss_steps, [10, 20])
            np.testing.assert_almost_equal(
                valid_loss_values,
                [-1, -2]
            )


if __name__ == '__main__':
    unittest.main()
