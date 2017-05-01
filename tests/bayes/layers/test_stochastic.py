# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf

from tfsnippet import bayes, distributions
from tests.helper import TestCase


class StochasticTestCase(TestCase):

    def test_Normal(self):
        with tf.Graph().as_default():
            normal = bayes.Normal(0., 1.)
            self.assertIsInstance(normal.distribution, distributions.Normal)
            # assert the distribution scope is contained in the scope
            # of StochasticTensor
            self.assertEqual(normal.distribution.variable_scope.name,
                             'normal/distribution')


if __name__ == '__main__':
    unittest.main()
