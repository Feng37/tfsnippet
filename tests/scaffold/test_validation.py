# -*- coding: utf-8 -*-
import os
import unittest

import tensorflow as tf

from tfsnippet.scaffold import early_stopping
from tfsnippet.utils import (TemporaryDirectory,
                             set_variable_values,
                             get_variable_values)
from tests.helper import TestCase


def _populate_variables():
    a = tf.get_variable('a', shape=(), dtype=tf.int32)
    b = tf.get_variable('b', shape=(), dtype=tf.int32)
    c = tf.get_variable('c', shape=(), dtype=tf.int32)
    set_variable_values([a, b, c], [1, 2, 3])
    return [a, b, c]


class EarlyStoppingTestCase(TestCase):

    def test_basic(self):
        with self.get_session():
            a, b, c = _populate_variables()
            self.assertEqual(get_variable_values([a, b, c]), [1, 2, 3])

            # test: param-vars must not be empty
            with self.assertRaisesRegex(
                    ValueError, '`param_vars` must not be empty.'):
                with early_stopping([]):
                    pass

            # test: early-stopping context without updating loss
            with early_stopping([a, b]):
                set_variable_values([a], [10])
            self.assertEqual(get_variable_values([a, b, c]), [10, 2, 3])

            # test: the first loss will always cause saving
            with early_stopping([a, b]) as es:
                set_variable_values([a], [10])
                self.assertTrue(es.update(1.))
                set_variable_values([a, b], [100, 20])
            self.assertAlmostEqual(es.best_metric, 1.)
            self.assertEqual(get_variable_values([a, b, c]), [10, 2, 3])

            # test: memorize the best loss
            set_variable_values([a, b, c], [1, 2, 3])
            self.assertEqual(get_variable_values([a, b, c]), [1, 2, 3])
            with early_stopping([a, b]) as es:
                set_variable_values([a], [10])
                self.assertTrue(es.update(1.))
                self.assertAlmostEqual(es.best_metric, 1.)
                set_variable_values([a, b], [100, 20])
                self.assertTrue(es.update(.5))
                self.assertAlmostEqual(es.best_metric, .5)
                set_variable_values([a, b, c], [1000, 200, 30])
                self.assertFalse(es.update(.8))
                self.assertAlmostEqual(es.best_metric, .5)
            self.assertAlmostEqual(es.best_metric, .5)
            self.assertEqual(get_variable_values([a, b, c]), [100, 20, 30])

            # test: initial_loss
            set_variable_values([a, b, c], [1, 2, 3])
            self.assertEqual(get_variable_values([a, b, c]), [1, 2, 3])
            with early_stopping([a, b], initial_metric=.6) as es:
                set_variable_values([a], [10])
                self.assertFalse(es.update(1.))
                self.assertAlmostEqual(es.best_metric, .6)
                set_variable_values([a, b], [100, 20])
                self.assertTrue(es.update(.5))
                self.assertAlmostEqual(es.best_metric, .5)
            self.assertEqual(get_variable_values([a, b, c]), [100, 20, 3])

    def test_restore_on_error(self):
        with self.get_session():
            a, b, c = _populate_variables()
            self.assertEqual(get_variable_values([a, b, c]), [1, 2, 3])

            # test: do not restore on error
            with self.assertRaisesRegex(ValueError, 'value error'):
                with early_stopping([a, b], restore_on_error=False) as es:
                    self.assertTrue(es.update(1.))
                    set_variable_values([a, b], [10, 20])
                    raise ValueError('value error')
            self.assertAlmostEqual(es.best_metric, 1.)
            self.assertEqual(get_variable_values([a, b, c]), [10, 20, 3])

            # test: restore on error
            set_variable_values([a, b, c], [1, 2, 3])
            self.assertEqual(get_variable_values([a, b, c]), [1, 2, 3])
            with self.assertRaisesRegex(ValueError, 'value error'):
                with early_stopping([a, b], restore_on_error=True) as es:
                    self.assertTrue(es.update(1.))
                    set_variable_values([a, b], [10, 20])
                    raise ValueError('value error')
            self.assertAlmostEqual(es.best_metric, 1.)
            self.assertEqual(get_variable_values([a, b, c]), [1, 2, 3])

    def test_bigger_is_better(self):
        with self.get_session():
            a, b, c = _populate_variables()
            self.assertEqual(get_variable_values([a, b, c]), [1, 2, 3])

            # test: memorize the best loss
            set_variable_values([a, b, c], [1, 2, 3])
            self.assertEqual(get_variable_values([a, b, c]), [1, 2, 3])
            with early_stopping([a, b], smaller_is_better=False) as es:
                set_variable_values([a], [10])
                self.assertTrue(es.update(.5))
                self.assertAlmostEqual(es.best_metric, .5)
                set_variable_values([a, b], [100, 20])
                self.assertTrue(es.update(1.))
                self.assertAlmostEqual(es.best_metric, 1.)
                set_variable_values([a, b, c], [1000, 200, 30])
                self.assertFalse(es.update(.8))
                self.assertAlmostEqual(es.best_metric, 1.)
            self.assertAlmostEqual(es.best_metric, 1.)
            self.assertEqual(get_variable_values([a, b, c]), [100, 20, 30])

    def test_save_dir(self):
        with self.get_session():
            a, b, c = _populate_variables()
            self.assertEqual(get_variable_values([a, b, c]), [1, 2, 3])

            with TemporaryDirectory() as tempdir:
                # test cleanup save_dir
                save_dir = os.path.join(tempdir, '1')
                with early_stopping([a, b], save_dir=save_dir) as es:
                    self.assertTrue(es.update(1.))
                    self.assertTrue(
                        os.path.exists(os.path.join(save_dir, 'latest')))
                self.assertFalse(os.path.exists(save_dir))

                # test not cleanup save_dir
                save_dir = os.path.join(tempdir, '2')
                with early_stopping([a, b], save_dir=save_dir,
                                    cleanup=False) as es:
                    self.assertTrue(es.update(1.))
                    self.assertTrue(
                        os.path.exists(os.path.join(save_dir, 'latest')))
                self.assertTrue(
                    os.path.exists(os.path.join(save_dir, 'latest')))


if __name__ == '__main__':
    unittest.main()
