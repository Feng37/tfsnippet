# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf

from tests.helper import TestCase
from tfsnippet.components import Component, Sequential


class SequentialTestCase(TestCase):

    def test_construction(self):
        def f(inputs):
            return tf.get_variable('v', shape=()) * inputs

        class C(Component):

            def __init__(self, flag):
                self.flag = flag
                super(C, self).__init__()

            def _call(self, inputs):
                v = tf.get_variable('a' if self.flag else 'b', shape=())
                return v * inputs

        with tf.Graph().as_default():
            c = C(flag=True)
            seq = Sequential([
                f,
                c,
                tf.nn.relu,
                f,
                C(flag=False),
                c,
            ])
            self.assertEqual(tf.global_variables(), [])
            seq(tf.constant(1.))
            self.assertEqual(
                sorted(v.name for v in tf.global_variables()),
                ['c/__call__/a:0',
                 'c_1/__call__/b:0',
                 'sequential/__call__/layer0/v:0',
                 'sequential/__call__/layer3/v:0']
            )
            seq(tf.constant(2.))
            self.assertEqual(
                sorted(v.name for v in tf.global_variables()),
                ['c/__call__/a:0',
                 'c_1/__call__/b:0',
                 'sequential/__call__/layer0/v:0',
                 'sequential/__call__/layer3/v:0']
            )

    def test_errors(self):
        with tf.Graph().as_default():
            with self.assertRaisesRegex(
                    ValueError, '`components` must not be empty.'):
                Sequential([])


if __name__ == '__main__':
    unittest.main()
