# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.bayes import Normal
from tfsnippet.bayes.variational import sgvb
from tests.helper import TestCase


class LowerBoundTestCase(TestCase):

    def test_sgvb(self):
        with tf.Graph().as_default(), tf.Session().as_default():
            a = Normal(0., 1., observed=np.asarray([[0., 1., 2.]]))
            b = Normal(1., 2., observed=np.asarray([[1., 2., 3.]]))
            c = Normal(2., 3., observed=np.asarray([[2., 3., 4.]]))

            lower_bound = sgvb([a, b], [c])
            self.assertEqual(lower_bound.get_shape().as_list(), [1, 3])
            np.testing.assert_almost_equal(
                lower_bound.eval(),
                (a.log_prob() + b.log_prob() - c.log_prob()).eval()
            )

            lower_bound = sgvb([a], [b, c], axis=0)
            self.assertEqual(lower_bound.get_shape().as_list(), [3])
            np.testing.assert_almost_equal(
                lower_bound.eval(),
                (a.log_prob() - b.log_prob() - c.log_prob()).eval().reshape([3])
            )

            lower_bound = sgvb([a], [b, c], axis=[0, 1])
            self.assertEqual(lower_bound.get_shape().as_list(), [])
            np.testing.assert_almost_equal(
                lower_bound.eval(),
                np.mean((a.log_prob() - b.log_prob() - c.log_prob()).eval())
            )


if __name__ == '__main__':
    unittest.main()
