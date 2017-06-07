# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.bayes import BernoulliLayer, StochasticTensor
from tests.helper import TestCase


class BernoulliLayerTestCase(TestCase):

    def test_basic(self):
        layer = BernoulliLayer()
        output = layer({'logits': tf.zeros([10, 2])})
        self.assertIsInstance(output, StochasticTensor)
        with self.get_session():
            np.testing.assert_almost_equal(
                output.distribution.logits.eval(),
                np.zeros([10, 2])
            )
            np.testing.assert_almost_equal(
                output.distribution.probs.eval(),
                np.ones([10, 2]) * 0.5
            )


if __name__ == '__main__':
    unittest.main()
