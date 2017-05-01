# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.distributions import Bernoulli
from tests.distributions._helper import (get_distribution_samples,
                                         big_number_verify,
                                         compute_distribution_prob,
                                         compute_analytic_kld,
                                         N_SAMPLES)
from tests.helper import TestCase


class BernoulliTestCase(TestCase):
    TOL = dict(rtol=1e-3, atol=1e-5)
    LOGITS = np.asarray([0.0, 0.5, -1.0], dtype=np.float32)

    def test_construction_error(self):
        with tf.Graph().as_default(), tf.Session().as_default():
            # test construction due to data type error
            with self.assertRaisesRegex(
                    TypeError, 'Bernoulli distribution parameters must be '
                               'float numbers'):
                Bernoulli(1)

            # test construction error due to shape mismatch
            with self.assertRaisesRegex(
                    TypeError, '`dtype` is expected to be an integer type, '
                                'but got .*'):
                Bernoulli(1., tf.float32)

    def test_construction(self):
        tol, logits = self.TOL, self.LOGITS
        p = 1. / (1 + np.exp(-logits))

        with tf.Graph().as_default(), tf.Session().as_default():
            # test basic attributes
            dist = Bernoulli(logits)
            self.assertEqual(dist.dtype, tf.int32)
            self.assertEqual(dist.param_dtype, tf.float32)
            self.assertFalse(dist.is_continuous)

            dist = Bernoulli(logits.astype(np.float64), tf.int64)
            self.assertEqual(dist.dtype, tf.int64)
            self.assertEqual(dist.param_dtype, tf.float64)

            # test shape attributes with fully static parameters
            dist = Bernoulli(logits)
            self.assertEqual(dist.static_value_shape.as_list(), [])
            self.assertEqual(dist.dynamic_value_shape, ())
            self.assertEqual(dist.static_batch_shape.as_list(), [3])
            self.assertEqual(
                tuple(tf.convert_to_tensor(dist.dynamic_batch_shape).eval()),
                (3,)
            )

            # test shape attributes with dynamic batch size
            dist = Bernoulli(tf.placeholder(tf.float32, shape=(None, 2)))
            self.assertEqual(dist.static_value_shape.as_list(), [])
            self.assertEqual(dist.dynamic_value_shape, ())
            self.assertEqual(dist.static_batch_shape.as_list(), [None, 2])
            self.assertEqual(
                tuple(dist.dynamic_batch_shape.eval({
                    dist.logits: np.arange(6, dtype=np.float32).reshape((3, 2))
                })),
                (3, 2)
            )

            # test the parameters of the distribution
            dist = Bernoulli(logits)
            np.testing.assert_allclose(dist.logits.eval(), logits, **tol)
            np.testing.assert_allclose(dist.mean.eval(), p, **tol)
            np.testing.assert_allclose(dist.p.eval(), p, **tol)
            np.testing.assert_allclose(dist.p_take_zero.eval(), 1. - p, **tol)

    def test_sampling_and_prob(self):
        tol, logits = self.TOL, self.LOGITS
        p = 1. / (1 + np.exp(-logits))
        mean = p
        stddev = p * (1. - p)

        def likelihood(x, logits, group_event_ndims=None):
            log_p = -np.log(1. + np.exp(-logits))
            log_one_minus_p = -np.log(1. + np.exp(logits))
            log_prob = x * log_p + (1 - x) * log_one_minus_p
            prob = np.exp(log_prob)
            if group_event_ndims:
                grouped_shape = prob.shape[: -group_event_ndims] + (-1,)
                prob = np.prod(prob.reshape(grouped_shape), axis=-1)
                log_prob = np.sum(log_prob.reshape(grouped_shape), axis=-1)
            return np.asarray([prob, log_prob])

        # test 1d sampling
        with tf.Graph().as_default(), tf.Session().as_default() as session:
            dist = Bernoulli(logits)
            samples_tensor = dist.sample()
            self.assertEqual(samples_tensor.dtype, tf.int32)
            samples, prob, log_prob = session.run([
                samples_tensor,
                dist.prob(samples_tensor),
                dist.log_prob(samples_tensor)
            ])
            self.assertEqual(samples.shape, (3,))
            true_prob, true_log_prob = likelihood(samples, logits)
            np.testing.assert_allclose(prob, true_prob, **tol)
            np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test 2d sampling
        samples, prob, log_prob = get_distribution_samples(
            Bernoulli, {'logits': logits}
        )
        true_prob, true_log_prob = likelihood(samples, logits)
        self.assertEqual(samples.shape, (N_SAMPLES, 3))
        np.testing.assert_equal(sorted(np.unique(samples)), [0, 1])
        big_number_verify(np.mean(samples, axis=0), mean, stddev, N_SAMPLES,
                          scale=6.)
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test 2d sampling (explicit batch size)
        samples, prob, log_prob = get_distribution_samples(
            Bernoulli, {'logits': logits},
            explicit_batch_size=True
        )
        true_prob, true_log_prob = likelihood(samples, logits)
        self.assertEqual(samples.shape, (N_SAMPLES, 3))
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test extra sampling shape
        samples, prob, log_prob = get_distribution_samples(
            Bernoulli, {'logits': logits},
            sample_shape=(4, 5)
        )
        true_prob, true_log_prob = likelihood(samples, logits)
        self.assertEqual(samples.shape, (4, 5, N_SAMPLES, 3))
        big_number_verify(
            np.mean(samples.reshape([-1, 3]), axis=0), mean, stddev,
            N_SAMPLES * 20, scale=6.
        )
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test extra sampling shape (with explicit batch shape)
        samples, prob, log_prob = get_distribution_samples(
            Bernoulli, {'logits': logits},
            sample_shape=(4, 5), explicit_batch_size=True
        )
        true_prob, true_log_prob = likelihood(samples, logits)
        self.assertEqual(samples.shape, (4, 5, N_SAMPLES, 3))
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test extra sampling shape == 1
        samples, prob, log_prob = get_distribution_samples(
            Bernoulli, {'logits': logits},
            sample_shape=[1]
        )
        true_prob, true_log_prob = likelihood(samples, logits)
        self.assertEqual(samples.shape, (1, N_SAMPLES, 3))
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test 3d sampling
        bias = [[0.0], [-1.0], [2.0], [-3.0]]
        logits_3d = logits.reshape([1, 3]) + bias
        samples, prob, log_prob = get_distribution_samples(
            Bernoulli, {'logits': logits_3d}
        )
        true_prob, true_log_prob = likelihood(samples, logits_3d)
        self.assertEqual(samples.shape, (N_SAMPLES, 4, 3))
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test computing likelihood with `group_event_ndims` arg
        prob, log_prob = compute_distribution_prob(
            Bernoulli, {'logits': logits_3d}, samples,
            group_event_ndims=1
        )
        true_prob, true_log_prob = likelihood(samples, logits_3d,
                                              group_event_ndims=1)
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        prob, log_prob = compute_distribution_prob(
            Bernoulli, {'logits': logits_3d}, samples,
            group_event_ndims=2
        )
        true_prob, true_log_prob = likelihood(samples, logits_3d,
                                              group_event_ndims=2)
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test computing log-likelihood on 1d samples with 2d parameters
        samples_1d = samples[0, 0, :]
        prob, log_prob = compute_distribution_prob(
            Bernoulli, {'logits': logits_3d}, samples_1d,
        )
        true_prob, true_log_prob = likelihood(samples_1d, logits_3d)
        self.assertEqual(samples_1d.shape, (3,))
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

    def test_analytic_kld(self):
        tol, logits = self.TOL, self.LOGITS
        logits2 = np.asarray([1.0, -0.3, 2.0], dtype=np.float32)

        def bernoulli_kld(logits1, logits2):
            p = 1. / (1. + np.exp(-logits1))
            q = 1. / (1. + np.exp(-logits2))
            return (
                p * np.log(p) + (1 - p) * np.log(1 - p)
                - p * np.log(q) - (1 - p) * np.log(1 - q)
            )

        kld = compute_analytic_kld(
            Bernoulli,
            {'logits': logits},
            {'logits': logits2},
        )
        np.testing.assert_allclose(
            kld, bernoulli_kld(logits, logits2), **tol)


if __name__ == '__main__':
    unittest.main()
