# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.bayes import NormalLayer, StochasticTensor
from tests.helper import TestCase


class NormalLayerTestCase(TestCase):

    def test_basic(self):
        layer = NormalLayer()
        output = layer({'mean': tf.zeros([10, 2]), 'logstd': tf.zeros([10, 2])})
        self.assertIsInstance(output, StochasticTensor)
        with self.get_session():
            np.testing.assert_almost_equal(
                output.distribution.mean.eval(),
                np.zeros([10, 2])
            )
            np.testing.assert_almost_equal(
                output.distribution.stddev.eval(),
                np.ones([10, 2])
            )


if __name__ == '__main__':
    unittest.main()
