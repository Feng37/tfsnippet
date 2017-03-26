# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.scaffold import LossValidator, option_defaults


class ValidationTestCase(unittest.TestCase):

    def test_LossValidator(self):
        with tf.Graph().as_default():
            # test run without even an input
            validator = LossValidator([], tf.constant(123.0, dtype=tf.float32))
            with tf.Session() as sess:
                np.testing.assert_allclose(validator.run([]), 123.0)

            # test compute the loss in one pass
            array_ph = tf.placeholder(dtype=tf.float32, shape=[None])
            loss = tf.reduce_mean(array_ph)
            data = np.arange(100, dtype=np.float32)
            validator = LossValidator([array_ph], loss)
            with tf.Session() as sess:
                np.testing.assert_allclose(
                    validator.run([data]), np.average(data))
                # test to run with empty input
                np.testing.assert_allclose(validator.run([data[:0]]), 0.0)

            # test compute the loss in batches (divisible)
            with option_defaults(predict_batch_size=10):
                validator = LossValidator([array_ph], loss)
                with tf.Session() as sess:
                    np.testing.assert_allclose(
                        validator.run([data]), np.average(data))
                    # test with empty data
                    np.testing.assert_allclose(
                        validator.run([data[:0]]), 0.0)

            # test compute the loss in batches (non-divisible)
            with option_defaults(predict_batch_size=32):
                validator = LossValidator([array_ph], loss)
                with tf.Session() as sess:
                    np.testing.assert_allclose(
                        validator.run([data]), np.average(data))

            # test compute the loss in batches (divisible, one batch)
            with option_defaults(predict_batch_size=100):
                validator = LossValidator([array_ph], loss)
                with tf.Session() as sess:
                    np.testing.assert_allclose(
                        validator.run([data]), np.average(data))

            # test compute the loss in batches (non-divisible, one batch)
            with option_defaults(predict_batch_size=101):
                validator = LossValidator([array_ph], loss)
                with tf.Session() as sess:
                    np.testing.assert_allclose(
                        validator.run([data]), np.average(data))

            # test variable saving and restoring
            a = tf.get_variable('a', dtype=tf.float32, initializer=0.)
            a_ph = tf.placeholder(dtype=tf.float32, shape=(), name='a_ph')
            a_assign = tf.assign(a, a_ph)
            loss_ph = tf.placeholder(dtype=tf.float32, shape=(), name='loss_ph')
            validator = LossValidator([], loss_ph, managed_vars={'a': a})

            with tf.Session() as sess:
                with validator.keep_best():
                    sess.run(a_assign, feed_dict={a_ph: 1})
                    validator.run([], feed_dict={loss_ph: 1})
                    sess.run(a_assign, feed_dict={a_ph: 2})
                    validator.run([], feed_dict={loss_ph: -1})
                    sess.run(a_assign, feed_dict={a_ph: 3})
                    validator.run([], feed_dict={loss_ph: 100})
                self.assertEqual(sess.run(a), 2)

if __name__ == '__main__':
    unittest.main()
