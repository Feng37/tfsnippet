# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.bayes import (Normal, StochasticTensor, gather_log_lower_bound,
                             reduce_log_lower_bound)
from tests.helper import TestCase


class GatherLogLowerBoundTestCase(TestCase):

    def test_basic(self):
        with self.get_session() as sess:
            a = Normal(0., np.asarray([1., 2., 3.]),
                       group_event_ndims=1).sample_n(16)
            b = Normal(1., 2.).sample_n(16)
            a_prob, b_prob = gather_log_lower_bound([a, b])

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
            self.assertEqual(gather_log_lower_bound([]), ())


class ReduceLogLowerBoundTestCase(TestCase):

    def test_basic(self):
        with self.get_session():
            prob = np.arange(24, dtype=np.float32).reshape([3, 2, 4])
            np.testing.assert_almost_equal(
                reduce_log_lower_bound(prob, 0).eval(),
                prob
            )
            np.testing.assert_almost_equal(
                reduce_log_lower_bound(prob, 1).eval(),
                np.sum(prob, axis=-1)
            )
            np.testing.assert_almost_equal(
                reduce_log_lower_bound(prob, 2).eval(),
                np.sum(prob.reshape([3, -1]), axis=-1)
            )
            np.testing.assert_almost_equal(
                reduce_log_lower_bound(prob, 3).eval(),
                np.sum(prob)
            )


if __name__ == '__main__':
    unittest.main()
