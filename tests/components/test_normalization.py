# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf

from tfsnippet.components import BatchNorm
from tests.helper import TestCase


class BatchNormTestCase(TestCase):

    def test_Construction(self):
        with tf.Graph().as_default():
            layer = BatchNorm()
            inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
            _ = layer(inputs)
            _ = layer(inputs)

            self.assertEqual(
                sorted(v.name for v in tf.global_variables()),
                ['batch_norm/batch_normalization/beta:0',
                 'batch_norm/batch_normalization/gamma:0',
                 'batch_norm/batch_normalization/moving_mean:0',
                 'batch_norm/batch_normalization/moving_variance:0']
            )

if __name__ == '__main__':
    unittest.main()
