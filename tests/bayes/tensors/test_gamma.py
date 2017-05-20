# -*- coding: utf-8 -*-
import unittest

from tfsnippet import bayes, distributions
from tests.helper import TestCase
from tests.bayes.tensors._helper import DerivedTensorTestMixin


class GammaTestCase(TestCase, DerivedTensorTestMixin):

    tensor_class = bayes.Gamma
    distrib_class = distributions.Gamma
    simple_args = {'alpha': [1.0, 2.0, 3.0], 'beta': [1.0, 2.0, 3.0]}


if __name__ == '__main__':
    unittest.main()
