# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet import bayes
from tests.helper import TestCase


class UtilsTestCase(TestCase):

    def test_local_log_prob(self):
        with self.test_session() as sess:
            # test regular
            a = bayes.Normal(0., [1., 2., 3.], sample_num=16,
                             group_event_ndims=1)
            b = bayes.Normal(1., 2., sample_num=16)
            a_prob, b_prob = bayes.local_log_prob([a, b])

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

            # test local_log_prob on empty list
            self.assertEqual(bayes.local_log_prob([]), ())

if __name__ == '__main__':
    unittest.main()
