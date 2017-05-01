# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf

from tfsnippet.scaffold import (OptionDefaults, get_option_defaults,
                                option_defaults)
from tests.helper import TestCase


class OptionDefaultsTestCase(TestCase):

    def test_construction(self):
        with tf.Graph().as_default():
            defaults = OptionDefaults()
            self.assertIsInstance(
                defaults.get_optimizer(), tf.train.AdamOptimizer)
            self.assertIsNone(defaults.batch_size)

            defaults = OptionDefaults(batch_size=32)
            self.assertEqual(defaults.batch_size, 32)

            with self.assertRaises(AttributeError) as cm:
                defaults.batch_size = 33
            self.assertIn('OptionDefaults object is read-only',
                          str(cm.exception))

    def test_scoped_context(self):
        with tf.Graph().as_default():
            self.assertIsNone(get_option_defaults().batch_size)
            self.assertIsNone(get_option_defaults().predict_batch_size)

            with option_defaults(batch_size=32, predict_batch_size=512):
                self.assertEqual(get_option_defaults().batch_size, 32)
                self.assertEqual(get_option_defaults().predict_batch_size, 512)

                with option_defaults(batch_size=64):
                    self.assertEqual(get_option_defaults().batch_size, 64)
                    self.assertEqual(get_option_defaults().predict_batch_size,
                                     512)

                self.assertEqual(get_option_defaults().batch_size, 32)
                self.assertEqual(get_option_defaults().predict_batch_size, 512)

            self.assertIsNone(get_option_defaults().batch_size)
            self.assertIsNone(get_option_defaults().predict_batch_size)


if __name__ == '__main__':
    unittest.main()
