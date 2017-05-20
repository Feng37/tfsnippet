# -*- coding: utf-8 -*-
import unittest

from tfsnippet import bayes, distributions
from tests.helper import TestCase
from tests.bayes.tensors._helper import DerivedTensorTestMixin


class BernoulliTestCase(TestCase, DerivedTensorTestMixin):

    tensor_class = bayes.Bernoulli
    distrib_class = distributions.Bernoulli
    simple_args = {'logits': [0., 1., 2.]}


if __name__ == '__main__':
    unittest.main()
