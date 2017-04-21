# -*- coding: utf-8 -*-
import re
import unittest
from contextlib import contextmanager

import tensorflow as tf

from tfsnippet.utils import auto_reuse_variables


def _get_var(name, **kwargs):
    kwargs.setdefault('shape', ())
    return tf.get_variable(name, **kwargs)


def _get_op(name):
    return tf.add(1, 2, name=name)


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
        with tf.Graph().as_default() as graph1:
            # test reuse flag == False at the first time
            with auto_reuse_variables('a') as vs:
                self.assertEqual(vs.name, 'a')
                v1 = _get_var('v1')
                self.assertEqual(v1.name, 'a/v1:0')
                self.assertEqual(_get_op('op1').name, 'a/op1:0')

            # test reuse flag == True at the second time
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
                self.assertEqual(scope_b_a.name, 'b/a')
                self.assertIs(_get_var('v1'), b_v1)
                self.assertEqual(_get_var('v1').name, 'b/a/v1:0')
                self.assertEqual(_get_op('op1').name, 'b/a/op1_1:0')
                self._assert_var_not_exists('v2')

        with tf.Graph().as_default() as graph2:
            # test reuse the variable scope at different graph
            with auto_reuse_variables('a') as vs:
                self.assertEqual(vs.name, 'a')
                g2_v1 = _get_var('v1')
                self.assertIsNot(g2_v1, v1)
                self.assertEqual(g2_v1.name, 'a/v1:0')
                self.assertEqual(_get_op('op1').name, 'a/op1:0')

if __name__ == '__main__':
    unittest.main()
