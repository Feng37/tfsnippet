# -*- coding: utf-8 -*-
import re
import unittest
from contextlib import contextmanager

import tensorflow as tf

from tfsnippet.utils import auto_reuse_variables, static_reuse, instance_reuse


def _get_var(name, **kwargs):
    kwargs.setdefault('shape', ())
    return tf.get_variable(name, **kwargs)


def _get_op(name):
    return tf.add(1, 2, name=name)


class _Reusable(object):

    def __init__(self, name):
        with tf.variable_scope(None, default_name=name) as vs:
            self.variable_scope = vs

    @instance_reuse
    def f(self):
        return tf.get_variable('var', shape=()), tf.add(1, 2, name='op')

    @instance_reuse(scope='f')
    def f_1(self):
        return tf.get_variable('var', shape=())

    @instance_reuse(unique_name_scope=False)
    def g(self):
        return tf.get_variable('var', shape=()), tf.add(1, 2, name='op')


class ReuseTestCase(unittest.TestCase):

    @contextmanager
    def _assert_exception(self, exp, msg):
        with self.assertRaises(exp) as cm:
            yield
        got_msg = str(cm.exception)
        if hasattr(msg, 'match'):
            self.assertTrue(
                msg.match(got_msg),
                msg='expected message %r but got %r' % (msg, got_msg)
            )
        else:
            self.assertEquals(got_msg, msg)

    def _assert_var_not_exists(self, name):
        pattern = re.compile(r'^Variable (|.*/)%s does not exist.*' % name)
        with self._assert_exception(ValueError, pattern):
            _get_var(name)

    def test_auto_reuse_variables(self):
        with tf.Graph().as_default():
            # test reuse == False at the first time
            with auto_reuse_variables('a') as vs:
                self.assertEqual(vs.name, 'a')
                v1 = _get_var('v1')
                self.assertEqual(v1.name, 'a/v1:0')
                self.assertEqual(_get_op('op1').name, 'a/op1:0')

            # test reuse == True at the second time
            with auto_reuse_variables('a') as vs:
                self.assertEqual(vs.name, 'a')
                self.assertIs(_get_var('v1'), v1)
                self.assertEqual(_get_var('v1').name, 'a/v1:0')
                self.assertEqual(_get_op('op1').name, 'a/op1_1:0')
                self._assert_var_not_exists('v2')

            # test reuse the variable scope at different name
            with tf.variable_scope('b'):
                with auto_reuse_variables('a') as vs:
                    scope_b_a = vs
                    self.assertEqual(vs.name, 'b/a')
                    b_v1 = _get_var('v1')
                    self.assertIsNot(b_v1, v1)
                    self.assertEqual(b_v1.name, 'b/a/v1:0')
                    self.assertEqual(_get_op('op1').name, 'b/a/op1:0')

            # test to open the variable scope using scope object
            with auto_reuse_variables(scope_b_a) as vs:
                self.assertEqual(vs.name, 'b/a')
                self.assertIs(_get_var('v1'), b_v1)
                self.assertEqual(_get_var('v1').name, 'b/a/v1:0')
                self.assertEqual(_get_op('op1').name, 'b/a/op1_1:0')
                self._assert_var_not_exists('v2')

            # test to open the variable scope with unique name scope
            with auto_reuse_variables(scope_b_a, unique_name_scope=True) as vs:
                self.assertEqual(vs.name, 'b/a')
                self.assertIs(_get_var('v1'), b_v1)
                self.assertEqual(_get_var('v1').name, 'b/a/v1:0')
                self.assertEqual(_get_op('op1').name, 'b/a_1/op1:0')

            with auto_reuse_variables('c', unique_name_scope=True) as vs:
                self.assertEqual(vs.name, 'c')
                self.assertEqual(_get_var('v1').name, 'c/v1:0')
                self.assertEqual(_get_op('op1').name, 'c/op1:0')

            with auto_reuse_variables('c', unique_name_scope=True) as vs:
                self.assertEqual(vs.name, 'c')
                self.assertEqual(_get_var('v1').name, 'c/v1:0')
                self.assertEqual(_get_op('op1').name, 'c_1/op1:0')

        with tf.Graph().as_default():
            # test reuse the variable scope at different graph
            with auto_reuse_variables('a') as vs:
                self.assertEqual(vs.name, 'a')
                g2_v1 = _get_var('v1')
                self.assertIsNot(g2_v1, v1)
                self.assertEqual(g2_v1.name, 'a/v1:0')
                self.assertEqual(_get_op('op1').name, 'a/op1:0')

    def test_static_reuse(self):
        @static_reuse
        def f():
            var = tf.get_variable('var', shape=())
            op = tf.add(1, 2, name='op')
            return var, op

        @static_reuse(scope='f')
        def f_1():
            return tf.get_variable('var', shape=())

        @static_reuse(unique_name_scope=False)
        def g():
            var = tf.get_variable('var', shape=())
            op = tf.add(1, 2, name='op')
            return var, op

        with tf.Graph().as_default():
            # test reuse with default settings
            var_1, op_1 = f()
            var_2, op_2 = f()
            self.assertIs(var_1, var_2)
            self.assertIsNot(op_1, op_2)
            self.assertEqual(var_1.name, 'f/var:0')
            self.assertEqual(op_1.name, 'f/op:0')
            self.assertEqual(op_2.name, 'f_1/op:0')

            # test reuse according to the scope name
            var_3 = f_1()
            self.assertIs(var_3, var_1)

            # test reuse within different parent scope
            with tf.variable_scope('parent'):
                var_3, op_3 = f()
                self.assertIsNot(var_3, var_2)
                self.assertIsNot(op_3, op_2)
                self.assertEqual(var_3.name, 'parent/f/var:0')
                self.assertEqual(op_3.name, 'parent/f/op:0')

            # test reuse with `unique_name_scope == False`
            var_1, op_1 = g()
            var_2, op_2 = g()
            self.assertIs(var_1, var_2)
            self.assertIsNot(op_1, op_2)
            self.assertEqual(var_1.name, 'g/var:0')
            self.assertEqual(op_1.name, 'g/op:0')
            self.assertEqual(op_2.name, 'g/op_1:0')

    def test_instance_reuse(self):
        with tf.Graph().as_default():
            obj = _Reusable('obj')

            # test reuse with default settings
            var_1, op_1 = obj.f()
            var_2, op_2 = obj.f()
            self.assertIs(var_1, var_2)
            self.assertIsNot(op_1, op_2)
            self.assertEqual(var_1.name, 'obj/f/var:0')
            self.assertEqual(op_1.name, 'obj/f/op:0')
            self.assertEqual(op_2.name, 'obj/f_1/op:0')

            # test reuse according to the scope name
            var_3 = obj.f_1()
            self.assertIs(var_3, var_1)

            # test reuse within different parent scope
            with tf.variable_scope('parent'):
                var_3, op_3 = obj.f()
                self.assertIs(var_3, var_1)
                self.assertIsNot(op_3, op_1)
                self.assertEqual(var_3.name, 'obj/f/var:0')
                self.assertEqual(op_3.name, 'obj/f_3/op:0')

            # test reuse with `unique_name_scope == False`
            var_1, op_1 = obj.g()
            var_2, op_2 = obj.g()
            self.assertIs(var_1, var_2)
            self.assertIsNot(op_1, op_2)
            self.assertEqual(var_1.name, 'obj/g/var:0')
            self.assertEqual(op_1.name, 'obj/g/op:0')
            self.assertEqual(op_2.name, 'obj/g/op_1:0')

            # test reuse on another object
            obj2 = _Reusable('obj')
            var_1, op_1 = obj2.f()
            var_2, op_2 = obj2.f()
            self.assertIs(var_1, var_2)
            self.assertIsNot(op_1, op_2)
            self.assertEqual(var_1.name, 'obj_1/f/var:0')
            self.assertEqual(op_1.name, 'obj_1/f/op:0')
            self.assertEqual(op_2.name, 'obj_1/f_1/op:0')

if __name__ == '__main__':
    unittest.main()
