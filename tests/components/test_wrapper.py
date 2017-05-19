# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf

from tfsnippet.components import Linear
from tests.helper import TestCase


class WrapperTestCase(TestCase):

    def test_Linear(self):
        linear = Linear(100)
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        _ = linear(inputs)
        _ = linear(inputs)

        self.assertEqual(
            sorted(v.name for v in tf.global_variables()),
            ['linear/fully_connected/biases:0',
             'linear/fully_connected/weights:0']
        )

if __name__ == '__main__':
    unittest.main()
