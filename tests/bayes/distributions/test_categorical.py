# -*- coding: utf-8 -*-
import unittest

import numpy as np
import six
import tensorflow as tf

from tfsnippet.bayes import Categorical, OneHotCategorical
from tests.helper import TestCase
from tests.bayes.distributions._helper import (DistributionTestMixin,
                                               AnalyticKldTestMixin,
                                               big_number_verify)


def _softmax(x):
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, -1, keepdims=True)


def _log_softmax(x):
    return x - np.log(np.sum(np.exp(x), -1, keepdims=True))


class _CategoricalTestMixin(DistributionTestMixin,
                            AnalyticKldTestMixin):

    simple_params = {
        'logits': np.asarray(
            [[0.0, 0.5, 0.0], [-1.0, -0.5, 0.0], [-0.5, 0.5, 0.0],
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
            [[1.0, -0.5, 0.0], [0.0, 0.5, 0.0], [-1.0, -0.5, 0.0],
             [-0.5, 0.5, 0.0]],
            dtype=np.float32
        )
    }
    is_continuous = False
    is_reparameterized = False

    def get_dtype_for_param_dtype(self, param_dtype):
        return tf.int32

    def analytic_kld(self, params1, params2):
        p = _softmax(params1['logits'])
        log_p = _log_softmax(params1['logits'])
        log_q = _log_softmax(params2['logits'])
        return np.sum(p * (log_p - log_q), -1)

    big_number_samples = 10000
    big_number_scale = 5.0

    def test_construction_error(self):
        with self.get_session():
            with self.assertRaisesRegex(
                    ValueError, 'One and only one of `logits`, `probs` should '
                                'be specified.'):
                self.dist_class()

            with self.assertRaisesRegex(
                    ValueError, 'One and only one of `logits`, `probs` should '
                                'be specified.'):
                self.dist_class([1., 2.], [1., 2.])

            with self.assertRaisesRegex(
                    TypeError, 'Categorical distribution parameters must be '
                               'float numbers.'):
                self.dist_class([1, 2])

    def test_other_properties(self):
        with self.get_session():
            logits = self.simple_params['logits']
            probs = _softmax(logits)

            # test construction with logits
            dist = self.dist_class(logits=logits)
            self.assert_allclose(dist.logits.eval(), logits)
            self.assert_allclose(dist.probs.eval(), probs)
            self.assertEqual(dist.n_categories, 3)

            # test construction with probs
            dist = self.dist_class(probs=probs)
            self.assert_allclose(dist.logits.eval(), np.log(probs))
            self.assert_allclose(dist.probs.eval(), probs)
            self.assertEqual(dist.n_categories, 3)

            # test attributes of dynamic shape
            logits_ph = tf.placeholder(tf.float32, [None, None])
            dist = self.dist_class(logits=logits_ph)
            self.assertIsInstance(dist.n_categories, tf.Tensor)
            self.assertEqual(
                dist.n_categories.eval({
                    logits_ph: np.arange(12, dtype=np.float32).reshape([3, 4])
                }),
                4
            )

    def test_specify_data_type(self):
        dist = self.dist_class(dtype=tf.int64, **self.simple_params)
        self.assertEqual(dist.dtype, tf.int64)
        dist = self.dist_class(dtype='int32', **self.simple_params)
        self.assertEqual(dist.dtype, tf.int32)

    def test_sampling_with_probs(self):
        with self.get_session(use_gpu=True):
            params = {'probs': _softmax(self.simple_params['logits'])}
            x, prob, log_prob = self.get_samples_and_prob(**params)
            value_shape, batch_shape = \
                self.get_shapes_for_param(**self.simple_params)
            np.testing.assert_equal(x.shape, batch_shape + value_shape)
            self.assert_allclose(
                prob, self.prob(x, **self.simple_params))
            self.assert_allclose(
                log_prob, self.log_prob(x, **self.simple_params))

    def test_log_prob_with_float(self):
        with self.get_session():
            x, prob, log_prob = self.get_samples_and_prob(**self.simple_params)
            x = x.astype(np.float32)
            dist = self.dist_class(**self.simple_params)
            self.assert_allclose(
                dist.prob(x).eval(), self.prob(x, **self.simple_params))
            self.assert_allclose(
                dist.log_prob(x).eval(), self.log_prob(x, **self.simple_params))

    def test_analytic_kld_with_probs(self):
        with self.get_session(use_gpu=True):
            params = {'probs': _softmax(self.simple_params['logits'])}
            kld_params = {'probs': _softmax(self.kld_simple_params['logits'])}
            self.assert_allclose(
                (self.dist_class(**params).
                 analytic_kld(self.dist_class(**kld_params))).eval(),
                self.analytic_kld(self.simple_params, self.kld_simple_params)
            )


class CategoricalTestCase(TestCase, _CategoricalTestMixin):

    dist_class = Categorical

    def get_shapes_for_param(self, **params):
        return (), params['logits'].shape[:-1]

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


class OneHotCategoricalTestCase(TestCase, _CategoricalTestMixin):

    dist_class = OneHotCategorical

    def get_shapes_for_param(self, **params):
        return params['logits'].shape[-1:], params['logits'].shape[:-1]

    def log_prob(self, x, group_event_ndims=None, **params):
        logits = params['logits']
        x = x.astype(logits.dtype)
        log_p = _log_softmax(logits)
        log_prob = np.sum(x * log_p, -1)
        if group_event_ndims:
            grouped_shape = log_prob.shape[: -group_event_ndims] + (-1,)
            log_prob = np.sum(log_prob.reshape(grouped_shape), axis=-1)
        return log_prob

    def test_boundary_values_of_probs(self):
        with self.get_session():
            dist = self.dist_class(probs=[0., 1.])
            np.testing.assert_almost_equal(dist.probs.eval(), [0., 1.])
            np.testing.assert_almost_equal(dist.logits[1].eval(), 0.)
            np.testing.assert_almost_equal(dist.prob([1, 0]).eval(), 0.)
            np.testing.assert_almost_equal(dist.prob([0, 1]).eval(), 1.)
            np.testing.assert_almost_equal(dist.log_prob([0, 1]).eval(), 0.)


if __name__ == '__main__':
    unittest.main()
