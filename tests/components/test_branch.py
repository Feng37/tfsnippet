# -*- coding: utf-8 -*-
import unittest

import six
import tensorflow as tf

from tfsnippet.components import DictMapper, Linear, Dense
from tests.helper import TestCase


class DictMapperTestCase(TestCase):

    def test_construction(self):
        net = DictMapper({
            'a': Linear(2),
            'b': Dense(2),
            'c': lambda x: x * tf.get_variable('y', initializer=0.)
        })
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        output = net(inputs)
        self.assertIsInstance(output, dict)
        self.assertEqual(sorted(output.keys()), ['a', 'b', 'c'])
        for v in six.itervalues(output):
            self.assertIsInstance(v, tf.Tensor)

        _ = net(inputs)
        self.assertEqual(
            sorted(v.name for v in tf.global_variables()),
            ['dense/fully_connected/biases:0',
             'dense/fully_connected/weights:0',
             'dict_mapper/c/y:0',
             'linear/fully_connected/biases:0',
             'linear/fully_connected/weights:0']
        )

    def test_invalid_key(self):
        for k in ['.', '', '90ab', 'abc.def']:
            with self.assertRaisesRegex(
                ValueError, 'The key for `DictMapper` must be a valid '
                            'Python identifier.*'):
                _ = DictMapper({k: lambda x: x})


if __name__ == '__main__':
    unittest.main()
