# -*- coding: utf-8 -*-
import unittest

import numpy as np

from tfsnippet.bayes import Normal, StochasticTensor, sgvb
from tests.helper import TestCase


class SgvbTestCase(TestCase):

    def test_sgvb(self):
        with self.get_session():
            a = StochasticTensor(
                Normal(0., 1.), observed=np.asarray([[0., 1., 2.]]))
            b = StochasticTensor(
                Normal(1., 2.), observed=np.asarray([[1., 2., 3.]]))
            c = StochasticTensor(
                Normal(2., 3.), observed=np.asarray([[2., 3., 4.]]))

            lower_bound = sgvb([a, b], [c])
            self.assertEqual(lower_bound.get_shape().as_list(), [1, 3])
            np.testing.assert_almost_equal(
                lower_bound.eval(),
                (a.log_prob() + b.log_prob() - c.log_prob()).eval()
            )

            lower_bound = sgvb([a], [b, c], latent_axis=0)
            self.assertEqual(lower_bound.get_shape().as_list(), [3])
            np.testing.assert_almost_equal(
                lower_bound.eval(),
                (a.log_prob() - b.log_prob() - c.log_prob()).eval().reshape([3])
            )

            lower_bound = sgvb([a], [b, c], latent_axis=[0, 1])
            self.assertEqual(lower_bound.get_shape().as_list(), [])
            np.testing.assert_almost_equal(
                lower_bound.eval(),
                np.mean((a.log_prob() - b.log_prob() - c.log_prob()).eval())
            )


if __name__ == '__main__':
    unittest.main()
