# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.distributions import Normal
from tests.distributions._helper import (get_distribution_samples,
                                         big_number_verify,
                                         compute_distribution_prob,
                                         compute_analytic_kld,
                                         N_SAMPLES)
from tests.helper import TestCase


class NormalTestCase(TestCase):
    TOL = dict(rtol=1e-3, atol=1e-5)
    MEAN = np.asarray([0.0, 1.0, -2.0], dtype=np.float32)
    STDDEV = np.asarray([1.0, 2.0, 5.0], dtype=np.float32)

    def test_construction_error(self):
        with tf.Graph().as_default(), tf.Session().as_default():
            # test construction due to no std specified
            with self.assertRaisesRegex(
                    ValueError, 'At least one of `stddev`, `logstd` should be '
                                'specified'):
                Normal(1.)

            # test construction due to data type error
            with self.assertRaisesRegex(
                    TypeError, 'Normal distribution parameters must be float '
                               'numbers'):
                Normal(1, 2)

            # test construction error due to shape mismatch
            with self.assertRaisesRegex(
                    ValueError, '`mean` and `stddev`/`logstd` should be '
                                'broadcastable'):
                Normal(np.arange(2, dtype=np.float32),
                       np.arange(3, dtype=np.float32))

    def test_construction(self):
        tol, mean, stddev = self.TOL, self.MEAN, self.STDDEV

        with tf.Graph().as_default(), tf.Session().as_default():
            # test basic attributes
            dist = Normal(mean, stddev)
            self.assertEqual(dist.dtype, tf.float32)
            self.assertEqual(dist.param_dtype, tf.float32)
            self.assertTrue(dist.is_continuous)

            dist = Normal(mean.astype(np.float64), stddev)
            self.assertEqual(dist.dtype, tf.float64)
            self.assertEqual(dist.param_dtype, tf.float64)

            # test shape attributes with fully static parameters
            dist = Normal(mean, stddev)
            self.assertEqual(dist.static_value_shape.as_list(), [])
            self.assertEqual(dist.dynamic_value_shape, ())
            self.assertEqual(dist.static_batch_shape.as_list(), [3])
            self.assertEqual(
                tuple(dist.dynamic_batch_shape.eval()),
                (3,)
            )

            # test shape attributes with dynamic batch size
            dist = Normal(tf.placeholder(tf.float32, shape=(None, 2)), 1.)
            self.assertEqual(dist.static_value_shape.as_list(), [])
            self.assertEqual(dist.dynamic_value_shape, ())
            self.assertEqual(dist.static_batch_shape.as_list(), [None, 2])
            self.assertEqual(
                tuple(dist.dynamic_batch_shape.eval({
                    dist.mean: np.arange(6).reshape((3, 2))
                })),
                (3, 2)
            )

            # test the parameters of the distribution
            dist = Normal(mean, stddev=stddev)
            np.testing.assert_allclose(
                dist.mean.eval(), mean, **tol)
            np.testing.assert_allclose(
                dist.stddev.eval(), stddev, **tol)
            np.testing.assert_allclose(
                dist.logstd.eval(), np.log(stddev), **tol)
            np.testing.assert_allclose(
                dist.var.eval(), np.square(stddev), **tol)
            np.testing.assert_allclose(
                dist.logvar.eval(), 2. * np.log(stddev), **tol)
            np.testing.assert_allclose(
                dist.precision.eval(), 1. / np.square(stddev), **tol)
            np.testing.assert_allclose(
                dist.log_precision.eval(), -2. * np.log(stddev), **tol)

            # test the parameters of the distribution when logstd is specified
            dist = Normal(mean, logstd=np.log(stddev))
            np.testing.assert_allclose(
                dist.mean.eval(), mean, **tol)
            np.testing.assert_allclose(
                dist.stddev.eval(), stddev, **tol)
            np.testing.assert_allclose(
                dist.logstd.eval(), np.log(stddev), **tol)
            np.testing.assert_allclose(
                dist.var.eval(), np.square(stddev), **tol)
            np.testing.assert_allclose(
                dist.logvar.eval(), 2. * np.log(stddev), **tol)
            np.testing.assert_allclose(
                dist.precision.eval(), 1. / np.square(stddev), **tol)
            np.testing.assert_allclose(
                dist.log_precision.eval(), -2. * np.log(stddev), **tol)

    def test_sampling_and_prob(self):
        tol, mean, stddev = self.TOL, self.MEAN, self.STDDEV

        def likelihood(x, mu, std, group_event_ndims=None):
            var = std ** 2
            logstd = np.log(std)
            c = -0.5 * np.log(np.pi * 2)
            precision = 1. / var
            log_prob = c - logstd - 0.5 * precision * ((x - mu) ** 2)
            prob = (np.exp(-0.5 * (x - mu) ** 2 / std ** 2) /
                    np.sqrt(2 * np.pi) / std)
            if group_event_ndims:
                grouped_shape = prob.shape[: -group_event_ndims] + (-1,)
                prob = np.prod(prob.reshape(grouped_shape), axis=-1)
                log_prob = np.sum(log_prob.reshape(grouped_shape), axis=-1)
            return np.asarray([prob, log_prob])

        # test 1d sampling
        with tf.Graph().as_default(), tf.Session().as_default() as session:
            dist = Normal(mean, stddev)
            samples_tensor = dist.sample()
            self.assertEqual(samples_tensor.dtype, tf.float32)
            samples, prob, log_prob = session.run([
                samples_tensor,
                dist.prob(samples_tensor),
                dist.log_prob(samples_tensor)
            ])
            self.assertEqual(samples.shape, (3,))
            true_prob, true_log_prob = likelihood(samples, mean, stddev)
            np.testing.assert_allclose(prob, true_prob, **tol)
            np.testing.assert_allclose(log_prob, true_log_prob, **tol)

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
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test extra sampling shape == 1
        samples, prob, log_prob = get_distribution_samples(
            Normal, {'mean': mean, 'stddev': stddev},
            sample_shape=[1]
        )
        true_prob, true_log_prob = likelihood(samples, mean, stddev)
        self.assertEqual(samples.shape, (1, N_SAMPLES, 3))
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

        # test computing likelihood with `group_event_ndims` arg
        prob, log_prob = compute_distribution_prob(
            Normal, {'mean': mean_3d, 'stddev': stddev_3d}, samples,
            group_event_ndims=1
        )
        true_prob, true_log_prob = likelihood(samples, mean_3d, stddev_3d,
                                              group_event_ndims=1)
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        prob, log_prob = compute_distribution_prob(
            Normal, {'mean': mean_3d, 'stddev': stddev_3d}, samples,
            group_event_ndims=2
        )
        true_prob, true_log_prob = likelihood(samples, mean_3d, stddev_3d,
                                              group_event_ndims=2)
        np.testing.assert_allclose(prob, true_prob, **tol)
        np.testing.assert_allclose(log_prob, true_log_prob, **tol)

        # test computing log-likelihood on 1d samples with 2d parameters
        samples_1d = samples[0, 0, :]
        prob, log_prob = compute_distribution_prob(
            Normal, {'mean': mean_3d, 'stddev': stddev_3d}, samples_1d
        )
        true_prob, true_log_prob = likelihood(samples_1d, mean_3d, stddev_3d)
        self.assertEqual(samples_1d.shape, (3,))
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

    def test_analytic_kld(self):
        tol, mean, stddev = self.TOL, self.MEAN, self.STDDEV

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

        # test more complicated KL-divergence situations
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


if __name__ == '__main__':
    unittest.main()
