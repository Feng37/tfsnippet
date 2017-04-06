# -*- coding: utf-8 -*-
import unittest

import numpy as np

from tfsnippet.distribution import Normal
from .helper import (get_distribution_samples, N_SAMPLES, big_number_verify,
                     compute_distribution_prob)


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
