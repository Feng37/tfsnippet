import unittest

import tensorflow as tf

from tfsnippet.components import Component
from tests.helper import TestCase


class _MyComponent(Component):

    def _call(self):
        with tf.variable_scope(None, default_name='a'):
            v1 = tf.get_variable('var', shape=())
        with tf.variable_scope(None, default_name='a'):
            v2 = tf.get_variable('var', shape=())
        return v1, v2


class ComponentTestCase(TestCase):

    def test_Component(self):

        with tf.Graph().as_default():
            c = _MyComponent(name='comp')
            v1, v2 = c()
            self.assertEqual(v1.name, 'comp/a/var:0')
            self.assertEqual(v2.name, 'comp/a_1/var:0')
            v1_1, v2_1 = c()
            self.assertIs(v1_1, v1)
            self.assertIs(v2_1, v2)

            c2 = _MyComponent(name='comp')
            v1_2, v2_2 = c2()
            self.assertEqual(v1_2.name, 'comp/a/var:0')
            self.assertEqual(v2_2.name, 'comp/a_1/var:0')
            self.assertIs(v1_2, v1)
            self.assertIs(v2_2, v2)

            with tf.variable_scope('child'):
                v1_3, v2_3 = c()
                self.assertEqual(v1_3.name, 'comp/a/var:0')
                self.assertEqual(v2_3.name, 'comp/a_1/var:0')
                self.assertIs(v1_3, v1)
                self.assertIs(v2_3, v2)

            c3 = _MyComponent(default_name='comp')
            v1_4, v2_4 = c3()
            self.assertEqual(v1_4.name, 'comp_1/a/var:0')
            self.assertEqual(v2_4.name, 'comp_1/a_1/var:0')
            self.assertIsNot(v1_4, v1)
            self.assertIsNot(v2_4, v2)

            c4 = _MyComponent(default_name='comp')
            v1_5, v2_5 = c4()
            self.assertEqual(v1_5.name, 'comp_2/a/var:0')
            self.assertEqual(v2_5.name, 'comp_2/a_1/var:0')
            self.assertIsNot(v1_5, v1_4)
            self.assertIsNot(v2_5, v2_4)

if __name__ == '__main__':
    unittest.main()
