# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf

from tfsnippet.utils import VariableSaver, TemporaryDirectory, \
    try_get_variable_value


class SessionTestCase(unittest.TestCase):

    def test_try_get_variable_value(self):
        with tf.Graph().as_default():
            a = tf.get_variable('a', initializer=1, dtype=tf.int32)
            with tf.Session() as sess:
                self.assertIsNone(try_get_variable_value(a))
                sess.run(a.initializer)
                self.assertEqual(try_get_variable_value(a), 1)

    def test_VariableSaver(self):
        with tf.Graph().as_default():
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

                with tf.Session() as sess:
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
