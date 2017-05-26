# -*- coding: utf-8 -*-
import unittest

import numpy as np
import six
import tensorflow as tf

from tfsnippet.bayes import Bernoulli
from tests.helper import TestCase
from ._helper import (DistributionTestMixin,
                      BigNumberVerifyTestMixin,
                      AnalyticKldTestMixin,
                      EnumerableTestMixin)


class BernoulliTestCase(TestCase,
                        DistributionTestMixin,
                        BigNumberVerifyTestMixin,
                        AnalyticKldTestMixin,
                        EnumerableTestMixin):

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
    is_enumerable = True

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

    def get_enum_value_count_for_params(self, params):
        return 2

    def get_enum_samples_for_params(self, params):
        x = np.asarray([0, 1], dtype=np.int32)
        x = x.reshape([2] + [1] * len(params['logits'].shape))
        x = np.tile(x, (1,) + params['logits'].shape)
        return x

    # test cases for Bernoulli distribution
    def test_construction_error(self):
        with self.get_session():
            # test construction due to data type error
            with self.assertRaisesRegex(
                    TypeError, 'Bernoulli distribution parameters must be '
                               'float numbers'):
                Bernoulli(1)

            # test construction with no parameter
            with self.assertRaisesRegex(
                    ValueError, 'One and only one of `logits`, `probs` should '
                                'be specified.'):
                Bernoulli()

            with self.assertRaisesRegex(
                    ValueError, 'One and only one of `logits`, `probs` should '
                                'be specified.'):
                Bernoulli(1., 2.)

    def test_other_properties(self):
        logits = self.simple_params['logits']
        probs, _ = self.get_mean_stddev(logits=logits)

        with self.get_session():
            # test construction with logits
            dist = Bernoulli(logits=logits)
            self.assert_allclose(dist.logits.eval(), logits)
            self.assert_allclose(dist.mean.eval(), probs)
            self.assert_allclose(dist.probs.eval(), probs)

            # test construction with probs
            dist = Bernoulli(probs=probs)
            self.assert_allclose(dist.logits.eval(), logits)
            self.assert_allclose(dist.mean.eval(), probs)
            self.assert_allclose(dist.probs.eval(), probs)

    def test_specify_data_type(self):
        dist = Bernoulli(dtype=tf.int64, **self.simple_params)
        self.assertEqual(dist.dtype, tf.int64)
        dist = Bernoulli(dtype='int32', **self.simple_params)
        self.assertEqual(dist.dtype, tf.int32)

    def test_sampling_with_probs(self):
        logits = self.simple_params['logits']
        probs, _ = self.get_mean_stddev(logits=logits)

        with self.get_session(use_gpu=True):
            params = {'probs': probs}
            x, prob, log_prob = self.get_samples_and_prob(**params)
            value_shape, batch_shape = \
                self.get_shapes_for_param(**self.simple_params)
            np.testing.assert_equal(x.shape, batch_shape + value_shape)
            self.assert_allclose(
                prob, self.prob(x, **self.simple_params))
            self.assert_allclose(
                log_prob, self.log_prob(x, **self.simple_params))

    def test_analytic_kld_with_probs(self):
        logits = self.simple_params['logits']
        probs, _ = self.get_mean_stddev(logits=logits)
        kl_logits = self.kld_simple_params['logits']
        kl_probs, _ = self.get_mean_stddev(logits=kl_logits)

        with self.get_session(use_gpu=True):
            params = {'probs': probs}
            kld_params = {'probs': kl_probs}
            self.assert_allclose(
                (self.dist_class(**params).
                 analytic_kld(self.dist_class(**kld_params))).eval(),
                self.analytic_kld(self.simple_params, self.kld_simple_params)
            )


if __name__ == '__main__':
    unittest.main()
