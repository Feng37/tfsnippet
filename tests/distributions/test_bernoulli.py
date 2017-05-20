# -*- coding: utf-8 -*-
import unittest

import six
import numpy as np
import tensorflow as tf

from tfsnippet.distributions import Bernoulli
from tests.distributions._helper import (DistributionTestMixin,
                                         BigNumberVerifyTestMixin,
                                         AnalyticKldTestMixin)
from tests.helper import TestCase


class BernoulliTestCase(TestCase,
                        DistributionTestMixin,
                        BigNumberVerifyTestMixin,
                        AnalyticKldTestMixin):

    dist_class = Bernoulli
    simple_params = {
        'logits': np.asarray([0.0, 0.7, -0.7], dtype=np.float32)
    }
    extended_dimensional_params = {
        k: v + np.asarray([[0.0], [-0.5], [0.5], [1.0], [-1.0]],
                          dtype=np.float32)
        for k, v in six.iteritems(simple_params)
    }
    kld_simple_params = {
        'logits': np.asarray([1.0, -0.3, 2.0], dtype=np.float32),
    }
    is_continuous = False
    is_reparameterized = True

    def get_shapes_for_param(self, **params):
        return (), params['logits'].shape

    def get_dtype_for_param_dtype(self, param_dtype):
        return tf.int32

    def log_prob(self, x, group_event_ndims=None, **params):
        logits = params['logits']
        log_p = -np.log(1. + np.exp(-logits))
        log_one_minus_p = -np.log(1. + np.exp(logits))
        log_prob = x * log_p + (1 - x) * log_one_minus_p
        if group_event_ndims:
            grouped_shape = log_prob.shape[: -group_event_ndims] + (-1,)
            log_prob = np.sum(log_prob.reshape(grouped_shape), axis=-1)
        return log_prob

    def get_mean_stddev(self, **params):
        p = 1. / (1 + np.exp(-params['logits']))
        return p, p * (1. - p)

    def analytic_kld(self, params1, params2):
        p = 1. / (1. + np.exp(-params1['logits']))
        q = 1. / (1. + np.exp(-params2['logits']))
        return (
            p * np.log(p) + (1 - p) * np.log(1 - p)
            - p * np.log(q) - (1 - p) * np.log(1 - q)
        )

    # test cases for Bernoulli distribution
    def test_construction_error(self):
        with self.get_session():
            # test construction due to data type error
            with self.assertRaisesRegex(
                    TypeError, 'Bernoulli distribution parameters must be '
                               'float numbers'):
                Bernoulli(1)

    def test_other_properties(self):
        mean, _ = self.get_mean_stddev(**self.simple_params)

        with self.get_session():
            dist = Bernoulli(**self.simple_params)
            self.assert_allclose(
                dist.logits.eval(), self.simple_params['logits'])
            self.assert_allclose(dist.mean.eval(), mean)
            self.assert_allclose(dist.p.eval(), mean)
            self.assert_allclose(dist.p_take_zero.eval(), 1. - mean)

    def test_specify_data_type(self):
        dist = Bernoulli(dtype=tf.int64, **self.simple_params)
        self.assertEqual(dist.dtype, tf.int64)
        dist = Bernoulli(dtype='int32', **self.simple_params)
        self.assertEqual(dist.dtype, tf.int32)


if __name__ == '__main__':
    unittest.main()
