# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf

from tfsnippet import bayes, distributions
from tests.helper import TestCase


class StochasticTestCase(TestCase):

    def test_Normal(self):
        dist = bayes.Normal(0., 1.)
        self.assertIsInstance(dist.distribution, distributions.Normal)
        # assert the distribution scope is contained in the scope
        # of StochasticTensor
        self.assertEqual(dist.distribution.variable_scope.name,
                         'normal/distribution')

    def test_Bernoulli(self):
        dist = bayes.Bernoulli(0.)
        self.assertIsInstance(dist.distribution, distributions.Bernoulli)
        # assert the distribution scope is contained in the scope
        # of StochasticTensor
        self.assertEqual(dist.distribution.variable_scope.name,
                         'bernoulli/distribution')


if __name__ == '__main__':
    unittest.main()
