# -*- coding: utf-8 -*-
import unittest

import numpy as np
from scipy.special import gamma, gammaln, digamma
import tensorflow as tf

from tfsnippet.distributions import Gamma
from tests.distributions._helper import (get_distribution_samples,
                                         big_number_verify,
                                         compute_distribution_prob,
                                         compute_analytic_kld,
                                         N_SAMPLES)
from tests.helper import TestCase

TOL = dict(rtol=1e-3, atol=1e-5)
ALPHA = np.asarray(
    [0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0, 2.0], dtype=np.float32)
BETA = np.asarray(
    [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float32)


def _likelihood(x, alpha, beta, group_event_ndims=None):
    prob = ((beta ** alpha) * (x ** (alpha - 1)) * np.exp(-beta * x) /
            gamma(alpha))
    log_prob = (alpha * np.log(beta) + (alpha - 1) * np.log(x) -
                beta * x - gammaln(alpha))
    if group_event_ndims:
        grouped_shape = prob.shape[: -group_event_ndims] + (-1,)
        prob = np.prod(prob.reshape(grouped_shape), axis=-1)
        log_prob = np.sum(log_prob.reshape(grouped_shape), axis=-1)
    return np.asarray([prob, log_prob])


class GammaTestCase(TestCase):

    def test_construction_error(self):
        with tf.Graph().as_default(), tf.Session().as_default():
            # test construction due to data type error
            with self.assertRaisesRegex(
                    TypeError, 'Gamma distribution parameters must be '
                               'float numbers'):
                Gamma(1, 2)

            # test construction error due to shape mismatch
            with self.assertRaisesRegex(
                    ValueError, '`alpha` and `beta` should be broadcastable'):
                Gamma(np.arange(2, dtype=np.float32),
                      np.arange(3, dtype=np.float32))

    def test_construction(self):
        with tf.Graph().as_default(), tf.Session().as_default():
            # test basic attributes
            dist = Gamma(ALPHA, BETA)
            self.assertEqual(dist.dtype, tf.float32)
            self.assertEqual(dist.param_dtype, tf.float32)
            self.assertTrue(dist.is_continuous)
            self.assertFalse(dist.is_reparameterized)

            dist = Gamma(ALPHA.astype(np.float64), ALPHA)
            self.assertEqual(dist.dtype, tf.float64)
            self.assertEqual(dist.param_dtype, tf.float64)

            # test shape attributes with fully static parameters
            dist = Gamma(ALPHA, BETA)
            self.assertEqual(dist.static_value_shape.as_list(), [])
            np.testing.assert_equal(dist.dynamic_value_shape.eval(), [])
            self.assertEqual(dist.static_batch_shape.as_list(), [9])
            self.assertEqual(
                tuple(dist.dynamic_batch_shape.eval()),
                (9,)
            )

            # test shape attributes with dynamic batch size
            dist = Gamma(tf.placeholder(tf.float32, shape=(None, 2)), 1.)
            self.assertEqual(dist.static_value_shape.as_list(), [])
            np.testing.assert_equal(dist.dynamic_value_shape.eval(), [])
            self.assertEqual(dist.static_batch_shape.as_list(), [None, 2])
            self.assertEqual(
                tuple(dist.dynamic_batch_shape.eval({
                    dist.alpha: np.arange(6).reshape((3, 2))
                })),
                (3, 2)
            )

            # test the parameters of the distribution
            dist = Gamma(ALPHA, BETA)
            np.testing.assert_allclose(
                dist.alpha.eval(), ALPHA, **TOL)
            np.testing.assert_allclose(
                dist.beta.eval(), BETA, **TOL)

    def test_sampling_and_prob(self):
        mean = ALPHA / BETA
        stddev = np.sqrt(ALPHA / (BETA ** 2))

        # test 1d sampling
        with tf.Graph().as_default(), tf.Session().as_default() as session:
            tf.set_random_seed(1234)
            dist = Gamma(ALPHA, BETA)
            samples_tensor = dist.sample()
            self.assertEqual(samples_tensor.dtype, tf.float32)
            samples, prob, log_prob = session.run([
                samples_tensor,
                dist.prob(samples_tensor),
                dist.log_prob(samples_tensor)
            ])
            self.assertEqual(samples.shape, (9,))
            true_prob, true_log_prob = _likelihood(samples, ALPHA, BETA)
            np.testing.assert_allclose(prob, true_prob, **TOL)
            np.testing.assert_allclose(log_prob, true_log_prob, **TOL)

            # test 2d sampling
            samples, prob, log_prob = get_distribution_samples(
                Gamma, {'alpha': ALPHA, 'beta': BETA}
            )
            true_prob, true_log_prob = _likelihood(samples, ALPHA, BETA)
            self.assertEqual(samples.shape, (N_SAMPLES, 9))
            big_number_verify(np.mean(samples, axis=0), mean, stddev, N_SAMPLES)
            np.testing.assert_allclose(prob, true_prob, **TOL)
            np.testing.assert_allclose(log_prob, true_log_prob, **TOL)

            # test 2d sampling (explicit batch size)
            samples, prob, log_prob = get_distribution_samples(
                Gamma, {'alpha': ALPHA, 'beta': BETA},
                explicit_batch_size=True
            )
            true_prob, true_log_prob = _likelihood(samples, ALPHA, BETA)
            self.assertEqual(samples.shape, (N_SAMPLES, 9))
            np.testing.assert_allclose(prob, true_prob, **TOL)
            np.testing.assert_allclose(log_prob, true_log_prob, **TOL)

            # test extra sampling shape
            samples, prob, log_prob = get_distribution_samples(
                Gamma, {'alpha': ALPHA, 'beta': BETA},
                sample_shape=(4, 5)
            )
            true_prob, true_log_prob = _likelihood(samples, ALPHA, BETA)
            self.assertEqual(samples.shape, (4, 5, N_SAMPLES, 9))
            big_number_verify(
                np.mean(samples.reshape([-1, 9]), axis=0), mean, stddev,
                N_SAMPLES * 20
            )
            np.testing.assert_allclose(prob, true_prob, **TOL)
            np.testing.assert_allclose(log_prob, true_log_prob, **TOL)

            # test extra sampling shape (with explicit batch shape)
            samples, prob, log_prob = get_distribution_samples(
                Gamma, {'alpha': ALPHA, 'beta': BETA},
                sample_shape=(4, 5), explicit_batch_size=True
            )
            true_prob, true_log_prob = _likelihood(samples, ALPHA, BETA)
            self.assertEqual(samples.shape, (4, 5, N_SAMPLES, 9))
            np.testing.assert_allclose(prob, true_prob, **TOL)
            np.testing.assert_allclose(log_prob, true_log_prob, **TOL)

            # test extra sampling shape == 1
            samples, prob, log_prob = get_distribution_samples(
                Gamma, {'alpha': ALPHA, 'beta': BETA},
                sample_shape=[1]
            )
            true_prob, true_log_prob = _likelihood(samples, ALPHA, BETA)
            self.assertEqual(samples.shape, (1, N_SAMPLES, 9))
            np.testing.assert_allclose(prob, true_prob, **TOL)
            np.testing.assert_allclose(log_prob, true_log_prob, **TOL)

            # test 3d sampling
            bias = [[0.0], [0.1], [0.2]]
            alpha_3d = ALPHA.reshape([1, 9]) + bias
            beta_3d = BETA.reshape([1, 9]) + bias
            mean_3d = alpha_3d / beta_3d
            stddev_3d = np.sqrt(alpha_3d / (beta_3d ** 2))
            samples, prob, log_prob = get_distribution_samples(
                Gamma, {'alpha': alpha_3d, 'beta': beta_3d}
            )
            true_prob, true_log_prob = _likelihood(samples, alpha_3d, beta_3d)
            self.assertEqual(samples.shape, (N_SAMPLES, 3, 9))
            for i in range(3):
                big_number_verify(
                    np.mean(samples[:, i, :], axis=0), mean_3d[i], stddev_3d[i],
                    N_SAMPLES
                )
            np.testing.assert_allclose(prob, true_prob, **TOL)
            np.testing.assert_allclose(log_prob, true_log_prob, **TOL)

            # test computing likelihood with `group_event_ndims` arg
            prob, log_prob = compute_distribution_prob(
                Gamma, {'alpha': alpha_3d, 'beta': beta_3d}, samples,
                group_event_ndims=1
            )
            true_prob, true_log_prob = _likelihood(samples, alpha_3d, beta_3d,
                                                   group_event_ndims=1)
            np.testing.assert_allclose(prob, true_prob, **TOL)
            np.testing.assert_allclose(log_prob, true_log_prob, **TOL)

            prob, log_prob = compute_distribution_prob(
                Gamma, {'alpha': alpha_3d, 'beta': beta_3d}, samples,
                group_event_ndims=2
            )
            true_prob, true_log_prob = _likelihood(samples, alpha_3d, beta_3d,
                                                   group_event_ndims=2)
            np.testing.assert_allclose(prob, true_prob, **TOL)
            np.testing.assert_allclose(log_prob, true_log_prob, **TOL)

            # test computing log-likelihood on 1d samples with 2d parameters
            samples_1d = samples[0, 0, :]
            prob, log_prob = compute_distribution_prob(
                Gamma, {'alpha': alpha_3d, 'beta': beta_3d}, samples_1d
            )
            true_prob, true_log_prob = _likelihood(samples_1d, alpha_3d,
                                                   beta_3d)
            self.assertEqual(samples_1d.shape, (9,))
            np.testing.assert_allclose(prob, true_prob, **TOL)
            np.testing.assert_allclose(log_prob, true_log_prob, **TOL)

    def test_analytic_kld(self):
        alpha2 = np.asarray(
            [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float32)
        beta2 = np.asarray(
            [0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0, 2.0], dtype=np.float32)

        def gamma_kld(alpha_1, beta_1, alpha_2, beta_2):
            return (
                (alpha_1 - alpha_2) * digamma(alpha_1) -
                gammaln(alpha_1) + gammaln(alpha_2) +
                alpha_2 * (np.log(beta_1) - np.log(beta_2)) +
                alpha_1 * (beta_2 / beta_1 - 1.)
            )

        kld = compute_analytic_kld(
            Gamma,
            {'alpha': ALPHA, 'beta': BETA},
            {'alpha': alpha2, 'beta': beta2},
        )
        np.testing.assert_allclose(
            kld, gamma_kld(ALPHA, BETA, alpha2, beta2), **TOL)


if __name__ == '__main__':
    unittest.main()
