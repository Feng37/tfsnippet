# -*- coding: utf-8 -*-
import unittest

from tfsnippet import bayes, distributions
from tests.helper import TestCase
from tests.bayes.tensors._helper import DerivedTensorTestMixin


class CategoricalTestCase(TestCase, DerivedTensorTestMixin):

    tensor_class = bayes.Categorical
    distrib_class = distributions.Categorical
    simple_args = {'logits': [[0., 1., 2.], [2., 1., 0.]]}


class OneHotCategoricalTestCase(TestCase, DerivedTensorTestMixin):

    tensor_class = bayes.OneHotCategorical
    distrib_class = distributions.OneHotCategorical
    simple_args = {'logits': [[0., 1., 2.], [2., 1., 0.]]}


if __name__ == '__main__':
    unittest.main()
