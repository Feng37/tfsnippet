# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet import bayes
from tests.helper import TestCase


class GatherLogProbTestCase(TestCase):

    def test_basic(self):
        with self.get_session() as sess:
            a = bayes.Normal(0., [1., 2., 3.], sample_num=16,
                             group_event_ndims=1)
            b = bayes.Normal(1., 2., sample_num=16)
            a_prob, b_prob = bayes.gather_log_prob([a, b])

            self.assertIsInstance(a_prob, tf.Tensor)
            self.assertEqual(a_prob.get_shape().as_list(), [16])
            res, ans = sess.run([
                a_prob,
                a.distribution.log_prob(a, group_event_ndims=1)
            ])
            np.testing.assert_almost_equal(res, ans)

            self.assertIsInstance(b_prob, tf.Tensor)
            self.assertEqual(b_prob.get_shape().as_list(), [16])
            res, ans = sess.run([
                b_prob,
                b.distribution.log_prob(b, group_event_ndims=0)
            ])
            np.testing.assert_almost_equal(res, ans)

    def test_empty(self):
        with self.get_session():
            self.assertEqual(bayes.gather_log_prob([]), ())


class LogProbReduceGroupEventsTestCase(TestCase):

    def test_basic(self):
        with self.get_session():
            prob = np.arange(24, dtype=np.float32).reshape([3, 2, 4])
            np.testing.assert_almost_equal(
                bayes.log_prob_reduce_group_events(prob, 0).eval(),
                prob
            )
            np.testing.assert_almost_equal(
                bayes.log_prob_reduce_group_events(prob, 1).eval(),
                np.sum(prob, axis=-1)
            )
            np.testing.assert_almost_equal(
                bayes.log_prob_reduce_group_events(prob, 2).eval(),
                np.sum(prob.reshape([3, -1]), axis=-1)
            )
            np.testing.assert_almost_equal(
                bayes.log_prob_reduce_group_events(prob, 3).eval(),
                np.sum(prob)
            )


if __name__ == '__main__':
    unittest.main()
