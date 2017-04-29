# -*- coding: utf-8 -*-
import unittest

import numpy as np

from tfsnippet.distributions import Normal
from .helper import (get_distribution_samples, big_number_verify,
                     compute_distribution_prob, compute_analytic_kld,
                     N_SAMPLES)


class NormalTestCase(unittest.TestCase):

    def test_diagonal_gaussian(self):
        tol = dict(rtol=1e-3, atol=1e-5)
        mean = np.asarray([0.0, 1.0, -2.0])
        stddev = np.asarray([1.0, 2.0, 5.0])

        def likelihood(x, mu, std):
            var = std ** 2
            logstd = np.log(std)
            c = -0.5 * np.log(np.pi * 2)
            precision = 1. / var
            log_prob = c - logstd - 0.5 * precision * ((x - mu) ** 2)
            prob = (np.exp(-0.5 * (x - mu) ** 2 / std ** 2) /
                    np.sqrt(2 * np.pi) / std)
            return np.asarray([prob, log_prob])

        # test 2d sampling
        samples, prob, log_prob = get_distribution_samples(
            Normal, {'mean': mean, 'stddev': stddev}
        )
        true_prob, true_log_prob = likelihood(samples, mean, stddev)
        self.assertEqual(samples.shape, (N_SAMPLES, 3))
        big_number_verify(np.mean(samples, axis=0), mean, stddev, N_SAMPLES)
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test 2d sampling (explicit batch size)
        samples, prob, log_prob = get_distribution_samples(
            Normal, {'mean': mean, 'stddev': stddev},
            explicit_batch_size=True
        )
        true_prob, true_log_prob = likelihood(samples, mean, stddev)
        self.assertEqual(samples.shape, (N_SAMPLES, 3))
        big_number_verify(np.mean(samples, axis=0), mean, stddev, N_SAMPLES)
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test extra sampling shape
        samples, prob, log_prob = get_distribution_samples(
            Normal, {'mean': mean, 'stddev': stddev},
            sample_shape=(4, 5)
        )
        true_prob, true_log_prob = likelihood(samples, mean, stddev)
        self.assertEqual(samples.shape, (4, 5, N_SAMPLES, 3))
        big_number_verify(
            np.mean(samples.reshape([-1, 3]), axis=0), mean, stddev,
            N_SAMPLES * 20
        )
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test extra sampling shape (with explicit batch shape)
        samples, prob, log_prob = get_distribution_samples(
            Normal, {'mean': mean, 'stddev': stddev},
            sample_shape=(4, 5), explicit_batch_size=True
        )
        true_prob, true_log_prob = likelihood(samples, mean, stddev)
        self.assertEqual(samples.shape, (4, 5, N_SAMPLES, 3))
        big_number_verify(
            np.mean(samples.reshape([-1, 3]), axis=0), mean, stddev,
            N_SAMPLES * 20
        )
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test extra sampling shape == 1
        samples, prob, log_prob = get_distribution_samples(
            Normal, {'mean': mean, 'stddev': stddev},
            sample_shape=[1]
        )
        true_prob, true_log_prob = likelihood(samples, mean, stddev)
        self.assertEqual(samples.shape, (1, N_SAMPLES, 3))
        big_number_verify(
            np.mean(samples.reshape([-1, 3]), axis=0), mean, stddev, N_SAMPLES)
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test 3d sampling
        bias = [[0.0], [3.0], [6.0], [9.0]]
        mean_3d = mean.reshape([1, 3]) + bias
        stddev_3d = stddev.reshape([1, 3]) + bias
        samples, prob, log_prob = get_distribution_samples(
            Normal, {'mean': mean_3d, 'stddev': stddev_3d}
        )
        true_prob, true_log_prob = likelihood(samples, mean_3d, stddev_3d)
        self.assertEqual(samples.shape, (N_SAMPLES, 4, 3))
        for i in range(4):
            big_number_verify(
                np.mean(samples[:, i, :], axis=0), mean + bias[i],
                stddev + bias[i], N_SAMPLES
            )
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test computing log-likelihood on 1d samples with 2d parameters
        sample_1d = samples[0, 0, :]
        prob, log_prob = compute_distribution_prob(
            Normal, {'mean': mean_3d, 'stddev': stddev_3d}, sample_1d
        )
        true_prob, true_log_prob = likelihood(sample_1d, mean_3d, stddev_3d)
        self.assertEqual(sample_1d.shape, (3,))
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test log sampling
        samples, prob, log_prob = get_distribution_samples(
            Normal, {'mean': mean, 'logstd': np.log(stddev)}
        )
        true_prob, true_log_prob = likelihood(samples, mean, stddev)
        self.assertEqual(samples.shape, (N_SAMPLES, 3))
        big_number_verify(np.mean(samples, axis=0), mean, stddev, N_SAMPLES)
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test the parameters of the distribution
        (x_prob, x_log_prob, x_mean, x_stddev, x_logstd, x_var, x_logvar,
         x_precision, x_log_precision) = \
            compute_distribution_prob(
                Normal, {'mean': mean, 'stddev': stddev}, mean,
                func=lambda d: [d.mean, d.stddev, d.logstd, d.var, d.logvar,
                                d.precision, d.log_precision]
            )

        np.testing.assert_allclose(x_mean, mean, **tol)
        np.testing.assert_allclose(x_stddev, stddev, **tol)
        np.testing.assert_allclose(x_logstd, np.log(stddev), **tol)
        np.testing.assert_allclose(x_var, np.square(stddev), **tol)
        np.testing.assert_allclose(x_logvar, 2. * np.log(stddev), **tol)
        np.testing.assert_allclose(x_precision, 1. / np.square(stddev), **tol)
        np.testing.assert_allclose(x_log_precision, -2. * np.log(stddev), **tol)

        # test the parameters of the distribution when logstd is specified
        (x_prob, x_log_prob, x_mean, x_stddev, x_logstd, x_var, x_logvar,
         x_precision, x_log_precision) = \
            compute_distribution_prob(
                Normal, {'mean': mean, 'logstd': np.log(stddev)}, mean,
                func=lambda d: [d.mean, d.stddev, d.logstd, d.var, d.logvar,
                                d.precision, d.log_precision]
            )

        np.testing.assert_allclose(x_mean, mean, **tol)
        np.testing.assert_allclose(x_stddev, stddev, **tol)
        np.testing.assert_allclose(x_logstd, np.log(stddev), **tol)
        np.testing.assert_allclose(x_var, np.square(stddev), **tol)
        np.testing.assert_allclose(x_logvar, 2. * np.log(stddev), **tol)
        np.testing.assert_allclose(x_precision, 1. / np.square(stddev), **tol)
        np.testing.assert_allclose(x_log_precision, -2. * np.log(stddev), **tol)

        # test KL-divergence defined by stddev
        kld = compute_analytic_kld(
            Normal,
            {'mean': mean, 'stddev': stddev},
            {'mean': np.asarray(0.), 'stddev': np.asarray(1.)}
        )
        logvar = 2. * np.log(stddev)
        np.testing.assert_allclose(
            kld,
            0.5 * (np.square(stddev) + np.square(mean) - 1 - logvar),
            **tol
        )

        # test KL-divergence defined by log-stddev
        kld = compute_analytic_kld(
            Normal,
            {'mean': mean, 'logstd': np.log(stddev)},
            {'mean': np.asarray(0.), 'logstd': np.asarray(0.)}
        )
        logvar = 2. * np.log(stddev)
        np.testing.assert_allclose(
            kld,
            0.5 * (np.square(stddev) + np.square(mean) - 1 - logvar),
            **tol
        )

        # test more complicated situations
        mean2 = np.asarray([-5.0, 3.0, 0.1])
        stddev2 = np.asarray([2.0, 0.1, 3.0])
        kld = compute_analytic_kld(
            Normal,
            {'mean': mean, 'stddev': stddev},
            {'mean': mean2, 'stddev': stddev2}
        )
        np.testing.assert_allclose(
            kld,
            0.5 * (
                np.square(stddev) / np.square(stddev2) +
                np.square(mean2 - mean) / np.square(stddev2) +
                2. * (np.log(stddev2) - np.log(stddev)) - 1
            ),
            **tol
        )
