import unittest

import tensorflow as tf

from tfsnippet.components import Component
from tests.helper import TestCase


class _MyComponent(Component):

    def _call(self):
        return tf.get_variable('var', shape=())


class ComponentTestCase(TestCase):

    def test_Component(self):

        with tf.Graph().as_default():
            c = _MyComponent(name='comp')
            v = c()
            self.assertEqual(v.name, 'comp/var:0')
            self.assertIs(c(), v)

            c2 = _MyComponent(name='comp')
            v2 = c2()
            self.assertEqual(v2.name, 'comp/var:0')
            self.assertIs(v2, v)

            with tf.variable_scope('child'):
                v_child = c()
                self.assertEqual(v_child.name, 'comp/var:0')
                self.assertIs(v_child, v)

            c3 = _MyComponent(default_name='comp')
            v3 = c3()
            self.assertEqual(v3.name, 'comp_1/var:0')
            self.assertIsNot(v3, v)

if __name__ == '__main__':
    unittest.main()
