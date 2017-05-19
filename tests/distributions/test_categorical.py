# -*- coding: utf-8 -*-
import unittest

import six
import numpy as np
import tensorflow as tf

from tfsnippet.distributions import Categorical
from tests.distributions._helper import (DistributionTestMixin,
                                         AnalyticKldTestMixin,
                                         big_number_verify)
from tests.helper import TestCase


def _softmax(x):
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, -1, keepdims=True)


def _log_softmax(x):
    return x - np.log(np.sum(np.exp(x), -1, keepdims=True))


class CategoricalTestCase(TestCase,
                          DistributionTestMixin,
                          AnalyticKldTestMixin):

    dist_class = Categorical
    simple_params = {
        'logits': np.asarray(
            [[0.0, 0.5, 1.0], [-1.0, -0.5, 0.0], [-0.5, 0.5, 0.0],
             [1.0, -0.5, 0.0]],
            dtype=np.float32
        )
    }
    extended_dimensional_params = {
        k: v + (
            np.asarray([0.0, -0.5, 0.5, 1.0, -1.0], dtype=np.float32).
            reshape([-1, 1, 1])
        )
        for k, v in six.iteritems(simple_params)
    }
    kld_simple_params = {
        'logits': np.asarray(
            [[1.0, -0.5, 0.0], [0.0, 0.5, 1.0], [-1.0, -0.5, 0.0],
             [-0.5, 0.5, 0.0]],
            dtype=np.float32
        )
    }
    is_continuous = False
    is_reparameterized = False

    def get_shapes_for_param(self, **params):
        return (), params['logits'].shape[:-1]

    def get_dtype_for_param_dtype(self, param_dtype):
        return tf.int32

    def log_prob(self, x, group_event_ndims=None, **params):
        logits = params['logits']
        x = x.astype(np.int32)
        x_one_hot = np.eye(logits.shape[-1])[x].astype(logits.dtype)
        log_p = _log_softmax(logits)
        log_prob = np.sum(x_one_hot * log_p, -1)
        if group_event_ndims:
            grouped_shape = log_prob.shape[: -group_event_ndims] + (-1,)
            log_prob = np.sum(log_prob.reshape(grouped_shape), axis=-1)
        return log_prob

    def analytic_kld(self, params1, params2):
        p = _softmax(params1['logits'])
        log_p = _log_softmax(params1['logits'])
        log_q = _log_softmax(params2['logits'])
        return np.sum(p * (log_p - log_q), -1)

    big_number_samples = 10000
    big_number_scale = 5.0

    # test cases for Bernoulli distribution
    def test_construction_error(self):
        with self.test_session():
            with self.assertRaisesRegex(
                    ValueError, 'One and only one of `logits`, `probs` should '
                                'be specified.'):
                Categorical()

            with self.assertRaisesRegex(
                    ValueError, 'One and only one of `logits`, `probs` should '
                                'be specified.'):
                Categorical([1., 2.], [1., 2.])

            with self.assertRaisesRegex(
                    TypeError, 'Categorical distribution parameters must be float '
                               'numbers.'):
                Categorical([1, 2])

    def test_other_properties(self):
        with self.test_session():
            logits = self.simple_params['logits']
            probs = _softmax(logits)

            # test construction with logits
            dist = Categorical(logits=logits)
            self.assert_allclose(dist.logits.eval(), logits)
            self.assert_allclose(dist.probs.eval(), probs)

            # test construction with probs
            dist = Categorical(probs=probs)
            self.assert_allclose(dist.logits.eval(), np.log(probs))
            self.assert_allclose(dist.probs.eval(), probs)

    def test_specify_data_type(self):
        dist = Categorical(dtype=tf.int64, **self.simple_params)
        self.assertEqual(dist.dtype, tf.int64)
        dist = Categorical(dtype='int32', **self.simple_params)
        self.assertEqual(dist.dtype, tf.int32)

    def test_log_prob_with_float(self):
        with self.test_session():
            x, prob, log_prob = self.get_samples_and_prob(**self.simple_params)
            x = x.astype(np.float32)
            dist = Categorical(**self.simple_params)
            self.assert_allclose(
                dist.prob(x).eval(), self.prob(x, **self.simple_params))
            self.assert_allclose(
                dist.log_prob(x).eval(), self.log_prob(x, **self.simple_params))

    def test_sampling_with_probs(self):
        with self.test_session(use_gpu=True):
            params = {'probs': _softmax(self.simple_params['logits'])}
            x, prob, log_prob = self.get_samples_and_prob(**params)
            value_shape, batch_shape = \
                self.get_shapes_for_param(**self.simple_params)
            np.testing.assert_equal(x.shape, batch_shape + value_shape)
            self.assert_allclose(
                prob, self.prob(x, **self.simple_params))
            self.assert_allclose(
                log_prob, self.log_prob(x, **self.simple_params))

    def test_analytic_kld_with_probs(self):
        with self.test_session(use_gpu=True):
            params = {'probs': _softmax(self.simple_params['logits'])}
            kld_params = {'probs': _softmax(self.kld_simple_params['logits'])}
            self.assert_allclose(
                (Categorical(**params).
                 analytic_kld(Categorical(**kld_params))).eval(),
                self.analytic_kld(self.simple_params, self.kld_simple_params)
            )

    def test_big_number_law(self):
        with self.test_session(use_gpu=True):
            logits = self.simple_params['logits']
            probs = _softmax(logits)
            x, prob, log_prob = self.get_samples_and_prob(
                sample_shape=[self.big_number_samples],
                **self.simple_params
            )
            for i in range(logits.shape[-1]):
                mean_i = probs[..., i]
                stddev_i = mean_i * (1. - mean_i)
                big_number_verify(
                    np.average(x == i, axis=0), mean_i, stddev_i,
                    self.big_number_scale, self.big_number_samples
                )


if __name__ == '__main__':
    unittest.main()
