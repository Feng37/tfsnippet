# -*- coding: utf-8 -*-
import re
import unittest
from contextlib import contextmanager

import tensorflow as tf

from tfsnippet.utils import open_variable_scope


def _get_var(name, **kwargs):
    kwargs.setdefault('shape', ())
    return tf.get_variable(name, **kwargs)


def _get_op(name):
    return tf.add(1, 2, name=name)


class ScopeUnitTest(unittest.TestCase):

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
            self.assertEqual(got_msg, msg)

    def _assert_var_exists(self, name):
        pattern = re.compile(r'^Variable (|.*/)%s already exists.*' % name)
        with self._assert_exception(ValueError, pattern):
            _get_var(name)

    def _assert_reuse_var(self, name):
        pattern = re.compile(
            r'^Trying to share variable (|.*/)%s, but specified shape.*' % name)
        with self._assert_exception(ValueError, pattern):
            _get_var(name, shape=(None,))

    def test_open_variable_scope(self):
        GLOBAL_VARIABLES_KEY = tf.GraphKeys.GLOBAL_VARIABLES

        with tf.Graph().as_default():
            # test to enter and re-enter sub scopes normally
            with open_variable_scope('a') as a:
                self.assertEqual(a.name, 'a')
                self.assertEqual(a.original_name_scope, 'a/')
                self.assertEqual(_get_var('v1').name, 'a/v1:0')
                self.assertEqual(_get_op('o1').name, 'a/o1:0')

                with open_variable_scope('b') as b:
                    self.assertEqual(b.name, 'a/b')
                    self.assertEqual(b.original_name_scope, 'a/b/')
                    self.assertEqual(_get_var('v1').name, 'a/b/v1:0')
                    self.assertEqual(_get_op('o1').name, 'a/b/o1:0')

                    with open_variable_scope(a) as a2:
                        self.assertEqual(a2.name, 'a')
                        self.assertEqual(a2.original_name_scope, 'a/')
                        self.assertEqual(_get_var('v2').name, 'a/v2:0')
                        self.assertEqual(_get_op('o2').name, 'a/o2:0')

                        with open_variable_scope('c') as c:
                            self.assertEqual(c.name, 'a/c')
                            self.assertEqual(c.original_name_scope, 'a/c/')
                            self.assertEqual(_get_var('v1').name, 'a/c/v1:0')
                            self.assertEqual(_get_op('o1').name, 'a/c/o1:0')

                a = tf.get_variable_scope()
                self.assertEqual(a.name, 'a')
                self.assertEqual(a.original_name_scope, 'a/')
                self.assertEqual(_get_var('v3').name, 'a/v3:0')
                self.assertEqual(_get_op('o3').name, 'a/o3:0')

            # test to enter sub scope with path
            with open_variable_scope('x/y/z') as xyz:
                self.assertEqual(xyz.name, 'x/y/z')
                self.assertEqual(xyz.original_name_scope, 'x/y/z/')
                xyz_v1 = _get_var('v1')
                self.assertEqual(xyz_v1.name, 'x/y/z/v1:0')
                self.assertEqual(_get_op('o1').name, 'x/y/z/o1:0')

            with open_variable_scope('x/y/w') as xyw:
                self.assertEqual(xyw.name, 'x/y/w')
                self.assertEqual(xyw.original_name_scope, 'x/y/w/')
                xyw_v1 = _get_var('v1')
                self.assertEqual(xyw_v1.name, 'x/y/w/v1:0')
                self.assertEqual(_get_op('o1').name, 'x/y/w/o1:0')

            self.assertEqual(
                tf.get_collection(GLOBAL_VARIABLES_KEY, scope='x'),
                [xyz_v1, xyw_v1]
            )
            self.assertEqual(
                tf.get_collection(GLOBAL_VARIABLES_KEY, scope='x/y'),
                [xyz_v1, xyw_v1]
            )
            self.assertEqual(
                tf.get_collection(GLOBAL_VARIABLES_KEY, scope='x/y/z'),
                [xyz_v1]
            )
            self.assertEqual(
                tf.get_collection(GLOBAL_VARIABLES_KEY, scope='x/y/w'),
                [xyw_v1]
            )

            # test to re-enter the root scope
            root = tf.get_variable_scope()
            self.assertEqual(root.name, '')
            self.assertEqual(root.original_name_scope, '')
            self.assertEqual(_get_var('v1').name, 'v1:0')
            self.assertEqual(_get_op('o1').name, 'o1:0')

            with open_variable_scope(xyz) as s:
                self.assertEqual(s.name, 'x/y/z')
                self.assertEqual(s.original_name_scope, 'x/y/z/')

                with open_variable_scope(root) as ss:
                    self.assertEqual(ss.name, '')
                    self.assertEqual(ss.original_name_scope, '')
                    self.assertEqual(_get_var('v2').name, 'v2:0')
                    self.assertEqual(_get_op('o2').name, 'o2:0')

                    with open_variable_scope(xyw) as sss:
                        self.assertEqual(sss.name, 'x/y/w')
                        self.assertEqual(sss.original_name_scope, 'x/y/w/')
                        self.assertEqual(_get_var('v2').name, 'x/y/w/v2:0')
                        self.assertEqual(_get_op('o2').name, 'x/y/w/o2:0')

                    ss = tf.get_variable_scope()
                    self.assertEqual(ss.name, '')
                    self.assertEqual(ss.original_name_scope, '')

                s = tf.get_variable_scope()
                self.assertEqual(s.name, 'x/y/z')
                self.assertEqual(s.original_name_scope, 'x/y/z/')

            # test to re-enter a deep scope.
            with open_variable_scope(c) as s:
                self.assertEqual(s.name, 'a/c')
                self.assertEqual(s.original_name_scope, 'a/c/')

                with open_variable_scope(xyz) as ss:
                    self.assertEqual(ss.name, 'x/y/z')
                    self.assertEqual(ss.original_name_scope, 'x/y/z/')
                    self.assertEqual(_get_var('v2').name, 'x/y/z/v2:0')
                    self.assertEqual(_get_op('o2').name, 'x/y/z/o2:0')

                self.assertEqual(s.name, 'a/c')
                self.assertEqual(s.original_name_scope, 'a/c/')

            # test to overwrite the scope settings.
            with open_variable_scope(c) as s:
                self.assertEqual(s.name, 'a/c')
                self.assertEqual(s.original_name_scope, 'a/c/')
                self._assert_var_exists('v1')
                self.assertEqual(_get_op('o1').name, 'a/c/o1_1:0')

                with open_variable_scope(s, reuse=True):
                    self.assertEqual(_get_var('v1').name, 'a/c/v1:0')
                    self.assertEqual(_get_op('o1').name, 'a/c/o1_2:0')
                    self._assert_reuse_var('v1')

            # test open an existing variable scope with unique name scope
            with open_variable_scope(a, unique_name_scope=True) as s:
                self.assertEqual(s.name, 'a')
                self.assertEqual(s.original_name_scope, 'a/')
                self._assert_var_exists('v1')
                self.assertEqual(_get_var('v4').name, 'a/v4:0')
                self.assertEqual(_get_op('o1').name, 'a_1/o1:0')

            with open_variable_scope(c, unique_name_scope=True) as s:
                self.assertEqual(s.name, 'a/c')
                self.assertEqual(s.original_name_scope, 'a/c/')
                self._assert_var_exists('v1')
                self.assertEqual(_get_var('v4').name, 'a/c/v4:0')
                self.assertEqual(_get_op('o1').name, 'a/c_1/o1:0')

            with open_variable_scope('a'):
                with open_variable_scope('c', unique_name_scope=True) as s:
                    self.assertEqual(s.name, 'a/c')
                    self.assertEqual(s.original_name_scope, 'a/c/')
                    self._assert_var_exists('v1')
                    self.assertEqual(_get_var('v5').name, 'a/c/v5:0')
                    self.assertEqual(_get_op('o1').name, 'a/c_2/o1:0')

            with open_variable_scope('d', unique_name_scope=True) as s:
                self.assertEqual(s.name, 'd')
                self.assertEqual(_get_op('o1').name, 'd/o1:0')

            with open_variable_scope('d', unique_name_scope=True) as s:
                self.assertEqual(s.name, 'd')
                self.assertEqual(_get_op('o1').name, 'd_1/o1:0')

            with self.assertRaises(ValueError):
                # root name scope cannot be opened via `unique_name_scope`.
                with open_variable_scope('', unique_name_scope=True) as s:
                    pass

            # test to open a pure variable scope
            with open_variable_scope(a) as s:
                with open_variable_scope('c', pure_variable_scope=True) as s:
                    self.assertEqual(s.name, 'a/c')
                    self.assertEqual(s.original_name_scope, 'a/c/')
                    self._assert_var_exists('v1')
                    self.assertEqual(_get_var('v6').name, 'a/c/v6:0')
                    self.assertEqual(_get_op('o4').name, 'a/o4:0')


if __name__ == '__main__':
    unittest.main()
