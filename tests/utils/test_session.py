# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf

from tfsnippet.utils import (VariableSaver,
                             TemporaryDirectory,
                             try_get_variable_value,
                             get_variable_values,
                             get_default_session_or_error,
                             get_uninitialized_variables,
                             set_variable_values)
from tests.helper import TestCase


class SessionTestCase(TestCase):

    def test_get_default_session_or_error(self):
        def do_raise():
            with self.assertRaises(RuntimeError) as cm:
                get_default_session_or_error()
            self.assertIn('No session is active.', str(cm.exception))

        do_raise()
        with self.get_session() as sess:
            self.assertIs(sess, get_default_session_or_error())
        do_raise()

    def test_try_get_variable_value(self):
        a = tf.get_variable('a', initializer=1, dtype=tf.int32)
        with self.get_session() as sess:
            self.assertIsNone(try_get_variable_value(a))
            sess.run(a.initializer)
            self.assertEqual(try_get_variable_value(a), 1)

    def test_get_uninitialized_variables(self):
        with self.get_session() as sess:
            a = tf.get_variable('a', dtype=tf.int32, initializer=1)
            b = tf.get_variable('b', dtype=tf.int32, initializer=2)
            c = tf.get_variable('c', dtype=tf.int32, initializer=3,
                                collections=[tf.GraphKeys.MODEL_VARIABLES])
            d = tf.get_variable('d', dtype=tf.int32, initializer=4,
                                collections=[tf.GraphKeys.MODEL_VARIABLES])
            self.assertEqual(
                get_uninitialized_variables(),
                [a, b]
            )
            self.assertEqual(
                get_uninitialized_variables([a, b, c, d]),
                [a, b, c, d]
            )
            sess.run(tf.variables_initializer([a, c]))
            self.assertEqual(
                get_uninitialized_variables(),
                [b]
            )
            self.assertEqual(
                get_uninitialized_variables([a, b, c, d]),
                [b, d]
            )
            sess.run(tf.variables_initializer([b, d]))
            self.assertEqual(
                get_uninitialized_variables(),
                []
            )
            self.assertEqual(
                get_uninitialized_variables([a, b, c, d]),
                []
            )

    def test_get_variable_values(self):
        with self.get_session() as sess:
            a = tf.get_variable('a', dtype=tf.int32, initializer=1)
            b = tf.get_variable('b', dtype=tf.int32, initializer=2)
            c = tf.get_variable('c', dtype=tf.int32, initializer=3)
            sess.run(tf.variables_initializer([a, b, c]))

            self.assertEqual(get_variable_values([a, b, c]),
                             [1, 2, 3])
            self.assertEqual(get_variable_values([a, b]),
                             [1, 2])
            self.assertEqual(get_variable_values({'a': a, 'b': b, 'c': c}),
                             {'a': 1, 'b': 2, 'c': 3})
            self.assertEqual(get_variable_values({'a': a, 'c': c}),
                             {'a': 1, 'c': 3})

    def test_set_variable_values(self):
        with self.get_session() as sess:
            a = tf.get_variable('a', dtype=tf.int32, initializer=1)
            b = tf.get_variable('b', dtype=tf.int32, initializer=2)
            c = tf.get_variable('c', dtype=tf.int32, initializer=3)

            set_variable_values([a, b], [10, 20])
            set_variable_values({'c': c}, {'c': 30})
            self.assertEqual(sess.run([a, b, c]), [10, 20, 30])

            sess.run(tf.variables_initializer([a, b, c]))
            self.assertEqual(sess.run([a, b, c]), [1, 2, 3])

            set_variable_values([a], [100])
            set_variable_values({'b': b, 'c': c}, {'b': 200, 'c': 300})
            self.assertEqual(sess.run([a, b, c]), [100, 200, 300])

            with self.assertRaises(TypeError):
                set_variable_values([a, b], {'a': 10, 'b': 20})
            with self.assertRaises(TypeError):
                set_variable_values({'a': a, 'b': b}, [10, 20])
            with self.assertRaises(TypeError):
                set_variable_values({'a': a, 'b': 20}, {'a': 10, 'b': 20})
            with self.assertRaises(IndexError):
                set_variable_values([a, b, c], [1, 2])
            with self.assertRaises(IndexError):
                set_variable_values([a, b], [1, 2, 3])
            with self.assertRaises(KeyError):
                set_variable_values({'a': a, 'b': b}, {'b': 20})

            set_variable_values({'a': a}, {'a': 10, 'b': 20})
            self.assertEqual(sess.run([a, b, c]), [10, 200, 300])

    def test_VariableSaver(self):
        a = tf.get_variable('a', initializer=1, dtype=tf.int32)
        b = tf.get_variable('b', initializer=2, dtype=tf.int32)
        c = tf.get_variable('c', initializer=3, dtype=tf.int32)
        a_ph = tf.placeholder(dtype=tf.int32, shape=(), name='a_ph')
        b_ph = tf.placeholder(dtype=tf.int32, shape=(), name='b_ph')
        c_ph = tf.placeholder(dtype=tf.int32, shape=(), name='c_ph')
        assign_op = tf.group(
            tf.assign(a, a_ph),
            tf.assign(b, b_ph),
            tf.assign(c, c_ph)
        )

        def get_values(sess):
            return sess.run([a, b, c])

        def set_values(sess, a, b, c):
            sess.run(assign_op, feed_dict={a_ph: a, b_ph: b, c_ph: c})

        with TemporaryDirectory() as tempdir1, \
                TemporaryDirectory() as tempdir2:
            saver1 = VariableSaver([a, b, c], tempdir1)
            saver2 = VariableSaver({'aa': a, 'bb': b}, tempdir2)

            with self.get_session() as sess:
                sess.run(tf.global_variables_initializer())
                self.assertEqual(get_values(sess), [1, 2, 3])
                set_values(sess, 10, 20, 30)
                self.assertEqual(get_values(sess), [10, 20, 30])

                saver1.save()
                set_values(sess, 100, 200, 300)
                self.assertEqual(get_values(sess), [100, 200, 300])
                saver1.restore()
                self.assertEqual(get_values(sess), [10, 20, 30])

                saver2.save()
                set_values(sess, 100, 200, 300)
                self.assertEqual(get_values(sess), [100, 200, 300])
                saver2.restore()
                self.assertEqual(get_values(sess), [10, 20, 300])

                saver1.restore()
                self.assertEqual(get_values(sess), [10, 20, 30])

                set_values(sess, 101, 201, 301)
                saver2.save()
                set_values(sess, 100, 200, 300)
                self.assertEqual(get_values(sess), [100, 200, 300])
                saver2.restore()
                self.assertEqual(get_values(sess), [101, 201, 300])

if __name__ == '__main__':
    unittest.main()
