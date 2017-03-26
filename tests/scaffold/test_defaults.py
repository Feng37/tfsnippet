# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf

from tfsnippet.scaffold import (OptionDefaults, get_option_defaults,
                                option_defaults)


class DefaultsTestCase(unittest.TestCase):

    def test_OptionDefaults(self):
        with tf.Graph().as_default():
            # test construction
            defaults = OptionDefaults()
            self.assertIsInstance(
                defaults.get_optimizer(), tf.train.AdamOptimizer)
            self.assertIsNone(defaults.batch_size)

            defaults = OptionDefaults(batch_size=32)
            self.assertEqual(defaults.batch_size, 32)

            # test write attribute
            with self.assertRaises(AttributeError):
                defaults.batch_size = 33

            # test scoped context
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
