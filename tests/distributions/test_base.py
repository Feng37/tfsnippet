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

    @property
    def is_enumerable(self):
        return False

    def _log_prob(self, x):
        return tf.reduce_sum(x * self.p, axis=-1)


class DistributionTestCase(TestCase):

    def _setUp(self):
        self.p_data = np.arange(1, 25, dtype=np.float32).reshape([2, 3, 4])
        self.p1 = tf.placeholder(tf.float32, (2, 3, None))
        self.p2 = tf.placeholder(tf.float32, (None, 3, 4))

        self.x_data = -(
            np.arange(1, 6, dtype=np.float32).reshape([5, 1, 1, 1]) +
            self.p_data.reshape([1, 2, 3, 4])
        )
        self.x1 = tf.placeholder(tf.float32, (5, 2, 3, None))
        self.x2 = tf.placeholder(tf.float32, (5, None, 3, 4))
        self.x3 = tf.placeholder(tf.float32, (None, 2, 3, 4))

        self.log_prob_data = np.sum(self.x_data * self.p_data, axis=-1)

    def test_group_event_ndims(self):
        with self.get_session():
            p_data = np.arange(1, 25, dtype=np.float32).reshape([2, 3, 4])

            # test the group_event_ndims attribute
            self.assertIsNone(_MyDistribution(p_data).group_event_ndims)
            self.assertEqual(_MyDistribution(p_data, 2).group_event_ndims, 2)
            with self.assertRaises(TypeError):
                _MyDistribution(p_data, -1)
            with self.assertRaises(TypeError):
                _MyDistribution(p_data, tf.placeholder(tf.int32, ()))

    def test_log_prob_with_static_data(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            log_prob = dist.log_prob(self.x_data)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2, 3])
            np.testing.assert_almost_equal(
                log_prob.eval(),
                self.log_prob_data
            )

            dist = _MyDistribution(self.p_data, group_event_ndims=1)
            log_prob = dist.log_prob(self.x_data)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2])
            np.testing.assert_almost_equal(
                log_prob.eval(),
                np.sum(self.log_prob_data, axis=-1)
            )

            log_prob = dist.log_prob(self.x_data, group_event_ndims=0)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2, 3])
            np.testing.assert_almost_equal(
                log_prob.eval(),
                self.log_prob_data
            )

    def test_log_prob_with_bad_shape(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            with self.assertRaisesRegex(
                    Exception, '.*Dimensions must be equal, but are 2 and 3.*'):
                dist.log_prob(self.x_data[:, :, :2, :])

            log_prob = dist.log_prob(self.x_data, group_event_ndims=2)
            self.assertEqual(log_prob.get_shape().as_list(), [5])
            np.testing.assert_almost_equal(
                log_prob.eval(),
                np.sum(np.sum(self.log_prob_data, axis=-1), axis=-1)
            )

    def test_log_prob_with_dynamic_value_shape(self):
        with self.get_session():
            dist = _MyDistribution(self.p1)
            log_prob = dist.log_prob(self.x1)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2, 3])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x1: self.x_data, self.p1: self.p_data}),
                self.log_prob_data
            )

            log_prob = dist.log_prob(self.x1, group_event_ndims=1)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x1: self.x_data, self.p1: self.p_data}),
                np.sum(self.log_prob_data, axis=-1)
            )

            log_prob = dist.log_prob(self.x1, group_event_ndims=2)
            self.assertEqual(log_prob.get_shape().as_list(), [5])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x1: self.x_data, self.p1: self.p_data}),
                np.sum(np.sum(self.log_prob_data, axis=-1), axis=-1)
            )

    def test_log_prob_with_dynamic_batch_shape(self):
        with self.get_session():
            dist = _MyDistribution(self.p2)
            log_prob = dist.log_prob(self.x2)
            self.assertEqual(log_prob.get_shape().as_list(), [5, None, 3])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x2: self.x_data, self.p2: self.p_data}),
                self.log_prob_data
            )

            log_prob = dist.log_prob(self.x2, group_event_ndims=1)
            self.assertEqual(log_prob.get_shape().as_list(), [5, None])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x2: self.x_data, self.p2: self.p_data}),
                np.sum(self.log_prob_data, axis=-1)
            )

            log_prob = dist.log_prob(self.x2, group_event_ndims=2)
            self.assertEqual(log_prob.get_shape().as_list(), [5])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x2: self.x_data, self.p2: self.p_data}),
                np.sum(np.sum(self.log_prob_data, axis=-1), axis=-1)
            )

    def test_log_prob_with_dynamic_data_and_batch_size(self):
        with self.get_session():
            dist = _MyDistribution(self.p2)
            log_prob = dist.log_prob(self.x3)
            # note: x3.shape == (None, 2, 3, 4), thus log_prob.shape[1] == 2
            self.assertEqual(log_prob.get_shape().as_list(), [None, 2, 3])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x3: self.x_data, self.p2: self.p_data}),
                self.log_prob_data
            )

            log_prob = dist.log_prob(self.x3, group_event_ndims=1)
            self.assertEqual(log_prob.get_shape().as_list(), [None, 2])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x3: self.x_data, self.p2: self.p_data}),
                np.sum(self.log_prob_data, axis=-1)
            )

            log_prob = dist.log_prob(self.x3, group_event_ndims=2)
            self.assertEqual(log_prob.get_shape().as_list(), [None])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x3: self.x_data, self.p2: self.p_data}),
                np.sum(np.sum(self.log_prob_data, axis=-1), axis=-1)
            )

    def test_prob_equal_to_exp_log_prob(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            prob = dist.prob(self.x_data[0, ...])
            np.testing.assert_allclose(
                prob.eval(),
                np.exp(self.log_prob_data[0, ...])
            )


if __name__ == '__main__':
    unittest.main()
