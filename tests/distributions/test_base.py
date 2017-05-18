# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.distributions import Distribution
from tfsnippet.utils import (reopen_variable_scope,
                             get_preferred_tensor_dtype,
                             get_dynamic_tensor_shape)
from tests.helper import TestCase


class _MyDistribution(Distribution):

    def __init__(self, p, group_event_ndims=None):
        super(_MyDistribution, self).__init__(
            group_event_ndims=group_event_ndims)

        with reopen_variable_scope(self.variable_scope):
            self.p = p = tf.convert_to_tensor(
                p, dtype=get_preferred_tensor_dtype(p))

            # get the shapes of parameter
            self._static_value_shape = p.get_shape()[-1:]
            self._dynamic_value_shape = tf.convert_to_tensor(
                get_dynamic_tensor_shape(p, lambda s: s[-1:])
            )

            self._static_batch_shape = p.get_shape()[: -1]
            self._dynamic_batch_shape = tf.convert_to_tensor(
                get_dynamic_tensor_shape(p, lambda s: s[: -1])
            )

    @property
    def dtype(self):
        return self.p.dtype.base_dtype

    @property
    def dynamic_value_shape(self):
        return self._dynamic_value_shape

    @property
    def static_value_shape(self):
        return self._static_value_shape

    @property
    def dynamic_batch_shape(self):
        return self._dynamic_batch_shape

    @property
    def static_batch_shape(self):
        return self._static_batch_shape

    def _log_prob(self, x):
        return tf.reduce_sum(x * self.p, axis=-1)


class DistributionTestCase(TestCase):

    def test_group_event_ndims(self):
        with tf.Graph().as_default(), tf.Session().as_default():
            p_data = np.arange(1, 25, dtype=np.float32).reshape([2, 3, 4])

            # test the group_event_ndims attribute
            self.assertIsNone(_MyDistribution(p_data).group_event_ndims)
            self.assertEqual(_MyDistribution(p_data, 2).group_event_ndims, 2)
            with self.assertRaises(TypeError):
                _MyDistribution(p_data, -1)
            with self.assertRaises(TypeError):
                _MyDistribution(p_data, tf.placeholder(tf.int32, ()))

    def test_log_prob(self):
        with tf.Graph().as_default(), tf.Session().as_default():
            p_data = np.arange(1, 25, dtype=np.float32).reshape([2, 3, 4])
            p1 = tf.placeholder(tf.float32, (2, 3, None))
            p2 = tf.placeholder(tf.float32, (None, 3, 4))

            x_data = -(
                np.arange(1, 6, dtype=np.float32).reshape([5, 1, 1, 1]) +
                p_data.reshape([1, 2, 3, 4])
            )
            x1 = tf.placeholder(tf.float32, (5, 2, 3, None))
            x2 = tf.placeholder(tf.float32, (5, None, 3, 4))
            x3 = tf.placeholder(tf.float32, (None, 2, 3, 4))

            log_prob_data = np.sum(x_data * p_data, axis=-1)

            # test log_prob with `group_event_ndims` by fully static data
            dist = _MyDistribution(p_data)
            log_prob = dist.log_prob(x_data)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2, 3])
            np.testing.assert_almost_equal(
                log_prob.eval(),
                log_prob_data
            )

            dist = _MyDistribution(p_data, group_event_ndims=1)
            log_prob = dist.log_prob(x_data)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2])
            np.testing.assert_almost_equal(
                log_prob.eval(),
                np.sum(log_prob_data, axis=-1)
            )

            log_prob = dist.log_prob(x_data, group_event_ndims=0)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2, 3])
            np.testing.assert_almost_equal(
                log_prob.eval(),
                log_prob_data
            )

            # assert bad shapes
            with self.assertRaisesRegex(
                    Exception, '.*Dimensions must be equal, but are 2 and 3.*'):
                dist.log_prob(x_data[:, :, :2, :])

            log_prob = dist.log_prob(x_data, group_event_ndims=2)
            self.assertEqual(log_prob.get_shape().as_list(), [5])
            np.testing.assert_almost_equal(
                log_prob.eval(),
                np.sum(np.sum(log_prob_data, axis=-1), axis=-1)
            )

            # test log_prob with `value_shape` dynamic
            dist = _MyDistribution(p1)
            log_prob = dist.log_prob(x1)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2, 3])
            np.testing.assert_almost_equal(
                log_prob.eval({x1: x_data, p1: p_data}),
                log_prob_data
            )

            log_prob = dist.log_prob(x1, group_event_ndims=1)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2])
            np.testing.assert_almost_equal(
                log_prob.eval({x1: x_data, p1: p_data}),
                np.sum(log_prob_data, axis=-1)
            )

            log_prob = dist.log_prob(x1, group_event_ndims=2)
            self.assertEqual(log_prob.get_shape().as_list(), [5])
            np.testing.assert_almost_equal(
                log_prob.eval({x1: x_data, p1: p_data}),
                np.sum(np.sum(log_prob_data, axis=-1), axis=-1)
            )

            # test log_prob with `batch_shape` dynamic
            dist = _MyDistribution(p2)
            log_prob = dist.log_prob(x2)
            self.assertEqual(log_prob.get_shape().as_list(), [5, None, 3])
            np.testing.assert_almost_equal(
                log_prob.eval({x2: x_data, p2: p_data}),
                log_prob_data
            )

            log_prob = dist.log_prob(x2, group_event_ndims=1)
            self.assertEqual(log_prob.get_shape().as_list(), [5, None])
            np.testing.assert_almost_equal(
                log_prob.eval({x2: x_data, p2: p_data}),
                np.sum(log_prob_data, axis=-1)
            )

            log_prob = dist.log_prob(x2, group_event_ndims=2)
            self.assertEqual(log_prob.get_shape().as_list(), [5])
            np.testing.assert_almost_equal(
                log_prob.eval({x2: x_data, p2: p_data}),
                np.sum(np.sum(log_prob_data, axis=-1), axis=-1)
            )

            # test log_prob with auxiliary dynamic data and dynamic
            # batch_size in parameter
            dist = _MyDistribution(p2)
            log_prob = dist.log_prob(x3)
            # note: x3.shape == (None, 2, 3, 4), thus log_prob.shape[1] == 2
            self.assertEqual(log_prob.get_shape().as_list(), [None, 2, 3])
            np.testing.assert_almost_equal(
                log_prob.eval({x3: x_data, p2: p_data}),
                log_prob_data
            )

            log_prob = dist.log_prob(x3, group_event_ndims=1)
            self.assertEqual(log_prob.get_shape().as_list(), [None, 2])
            np.testing.assert_almost_equal(
                log_prob.eval({x3: x_data, p2: p_data}),
                np.sum(log_prob_data, axis=-1)
            )

            log_prob = dist.log_prob(x3, group_event_ndims=2)
            self.assertEqual(log_prob.get_shape().as_list(), [None])
            np.testing.assert_almost_equal(
                log_prob.eval({x3: x_data, p2: p_data}),
                np.sum(np.sum(log_prob_data, axis=-1), axis=-1)
            )

            # assert that prob(x) == exp(log_prob(x))
            dist = _MyDistribution(p_data)
            prob = dist.prob(x_data[0, ...])
            np.testing.assert_allclose(
                prob.eval(),
                np.exp(log_prob_data[0, ...])
            )


if __name__ == '__main__':
    unittest.main()
