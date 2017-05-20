# -*- coding: utf-8 -*-
import unittest

from tfsnippet import bayes, distributions
from tests.helper import TestCase
from tests.bayes.tensors._helper import DerivedTensorTestMixin


class NormalTestCase(TestCase, DerivedTensorTestMixin):

    tensor_class = bayes.Normal
    distrib_class = distributions.Normal
    simple_args = {'mean': [1.0, 2.0, 3.0], 'stddev': [1.0, 2.0, 3.0]}


if __name__ == '__main__':
    unittest.main()
