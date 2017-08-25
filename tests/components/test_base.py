import unittest

import tensorflow as tf

from tfsnippet.components import Component, Lambda
from tests.helper import TestCase


class _MyComponent(Component):

    def _call(self):
        with tf.variable_scope(None, default_name='a'):
            v1 = tf.get_variable('var', shape=())
        with tf.variable_scope(None, default_name='a'):
            v2 = tf.get_variable('var', shape=())
        return v1, v2, tf.add(v1, v2, name='op')


class ComponentTestCase(TestCase):

    def test_construction_with_name(self):
        c = _MyComponent(name='comp')
        v1, v2, op = c()
        self.assertEqual(v1.name, 'comp/a/var:0')
        self.assertEqual(v2.name, 'comp/a_1/var:0')
        self.assertEqual(op.name, 'comp/build/op:0')
        v1_1, v2_1, op_1 = c()
        self.assertIs(v1_1, v1)
        self.assertIs(v2_1, v2)
        self.assertEqual(op_1.name, 'comp/build_1/op:0')

        c2 = _MyComponent(name='comp')
        v1_2, v2_2, op_2 = c2()
        self.assertEqual(v1_2.name, 'comp_1/a/var:0')
        self.assertEqual(v2_2.name, 'comp_1/a_1/var:0')
        self.assertEqual(op_2.name, 'comp_1/build/op:0')
        self.assertIsNot(v1_2, v1)
        self.assertIsNot(v2_2, v2)

    def test_construction_in_nested_scope(self):
        c = _MyComponent(name='comp')
        with tf.variable_scope('child'):
            v1, v2, op = c()
            self.assertEqual(v1.name, 'comp/a/var:0')
            self.assertEqual(v2.name, 'comp/a_1/var:0')
            self.assertEqual(op.name, 'comp/build/op:0')

    def test_construction_with_scope(self):
        c = _MyComponent(scope='comp')
        v1, v2, op = c()
        self.assertEqual(v1.name, 'comp/a/var:0')
        self.assertEqual(v2.name, 'comp/a_1/var:0')
        self.assertEqual(op.name, 'comp/build/op:0')

        c2 = _MyComponent(scope='comp')
        v1_1, v2_1, op_1 = c2()
        self.assertEqual(v1_1.name, 'comp/a/var:0')
        self.assertEqual(v2_1.name, 'comp/a_1/var:0')
        self.assertEqual(op_1.name, 'comp_1/build/op:0')
        self.assertIs(v1_1, v1)
        self.assertIs(v2_1, v2)


class LambdaTestCase(TestCase):

    def test_construction(self):
        def f(var_name, op_name):
            return (
                tf.get_variable(var_name, shape=(), dtype=tf.int32),
                tf.add(1, 2, name=op_name)
            )

        comp = Lambda(f)
        var, op = comp('var', 'op')
        self.assertEqual(var.name, 'lambda/var:0')
        self.assertEqual(op.name, 'lambda/build/op:0')
        var_1, op_1 = comp('var', 'op')
        self.assertEqual(var_1.name, 'lambda/var:0')
        self.assertEqual(op_1.name, 'lambda/build_1/op:0')
        self.assertIs(var_1, var)

if __name__ == '__main__':
    unittest.main()
