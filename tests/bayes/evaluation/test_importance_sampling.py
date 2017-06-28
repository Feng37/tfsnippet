#!/usr/bin/env python
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.bayes import Normal, importance_sampling
from tests.helper import TestCase


class ImportanceSamplingTestCase(TestCase):

    def test_element_wise(self):
        with self.get_session(use_gpu=True) as sess:
            tf.set_random_seed(1234)
            p = Normal(0., 1.)
            q = Normal(1., 2.)
            x = q.sample_n(1000)
            log_p = p.log_prob(x)
            log_q = q.log_prob(x)

            # test element-wise equality
            output, expected = sess.run(
                [importance_sampling(log_p, log_p, log_q),
                 log_p * tf.exp(log_p - log_q)]
            )
            np.testing.assert_almost_equal(output, expected)

    def test_integrated(self):
        with self.get_session(use_gpu=True) as sess:
            tf.set_random_seed(1234)
            p = Normal(0., 1.)
            q = Normal(1., 2.)
            x = q.sample_n(10000)
            log_p = p.log_prob(x)
            log_q = q.log_prob(x)

            # test integrated equality
            output = sess.run(
                importance_sampling(log_p, log_p, log_q, latent_axis=0))
            self.assertLess(np.abs(output + 0.5 * (np.log(2 * np.pi) + 1)),
                            1e-2)


if __name__ == '__main__':
    unittest.main()
