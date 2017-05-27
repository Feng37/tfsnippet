# -*- coding: utf-8 -*-
import copy
import unittest

import numpy as np
import six

from tfsnippet.bayes import Normal
from tests.helper import TestCase
from tests.bayes.distributions._helper import (DistributionTestMixin,
                                               BigNumberVerifyTestMixin,
                                               AnalyticKldTestMixin)


class NormalTestCase(TestCase,
                     DistributionTestMixin,
                     BigNumberVerifyTestMixin,
                     AnalyticKldTestMixin):

    dist_class = Normal
    simple_params = {
        'mean': np.asarray([0.0, 1.0, -2.0], dtype=np.float32),
        'stddev': np.asarray([1.0, 2.0, 5.0], dtype=np.float32),
    }
    extended_dimensional_params = {
        k: v + np.asarray([[0.0], [3.0], [6.0], [9.0]], dtype=np.float32)
        for k, v in six.iteritems(simple_params)
    }
    kld_simple_params = {
        'mean': np.asarray([-5.0, 3.0, 0.1], dtype=np.float32),
        'stddev': np.asarray([2.0, 0.1, 3.0], dtype=np.float32),
    }
    is_continuous = True
    is_reparameterized = True
    is_enumerable = False

    def get_shapes_for_param(self, **params):
        x = params['mean'] + params['stddev']
        return (), x.shape

    def log_prob(self, x, group_event_ndims=None, **params):
        mean, stddev = params['mean'], params['stddev']
        c = -0.5 * np.log(np.pi * 2)
        logstd = np.log(stddev)
        precision = np.exp(-2. * logstd)
        log_prob = c - logstd - 0.5 * precision * ((x - mean) ** 2)
        if group_event_ndims:
            grouped_shape = log_prob.shape[: -group_event_ndims] + (-1,)
            log_prob = np.sum(log_prob.reshape(grouped_shape), axis=-1)
        return log_prob

    def get_mean_stddev(self, **params):
        return params['mean'], params['stddev']

    def analytic_kld(self, params1, params2):
        mean1, stddev1 = params1['mean'], params1['stddev']
        mean2, stddev2 = params2['mean'], params2['stddev']
        return 0.5 * (
            np.square(stddev1) / np.square(stddev2) +
            np.square(mean2 - mean1) / np.square(stddev2) +
            2. * (np.log(stddev2) - np.log(stddev1)) - 1
        )

    # test cases for Normal distribution
    def test_construction_error(self):
        with self.get_session():
            # test construction due to no std specified
            with self.assertRaisesRegex(
                    ValueError, 'One and only one of `stddev`, `logstd` should '
                                'be specified.'):
                Normal(1.)

            with self.assertRaisesRegex(
                    ValueError, 'One and only one of `stddev`, `logstd` should '
                                'be specified.'):
                Normal(1., 2., 3.)

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

    def test_other_properties(self):
        with self.get_session():
            mean, stddev = self.get_mean_stddev(**self.simple_params)

            # test the parameters of the distribution
            dist = Normal(**self.simple_params)
            np.testing.assert_allclose(dist.mean.eval(), mean)
            np.testing.assert_allclose(dist.stddev.eval(), stddev)
            np.testing.assert_allclose(
                dist.logstd.eval(), np.log(stddev))
            np.testing.assert_allclose(
                dist.var.eval(), np.square(stddev))
            np.testing.assert_allclose(
                dist.logvar.eval(), 2. * np.log(stddev))
            np.testing.assert_allclose(
                dist.precision.eval(), 1. / np.square(stddev))
            np.testing.assert_allclose(
                dist.log_precision.eval(), -2. * np.log(stddev))

            # test the parameters of the distribution when logstd is specified
            dist = Normal(mean, logstd=np.log(stddev))
            np.testing.assert_allclose(dist.mean.eval(), mean)
            np.testing.assert_allclose(dist.stddev.eval(), stddev)
            np.testing.assert_allclose(
                dist.logstd.eval(), np.log(stddev))
            np.testing.assert_allclose(
                dist.var.eval(), np.square(stddev))
            np.testing.assert_allclose(
                dist.logvar.eval(), 2. * np.log(stddev))
            np.testing.assert_allclose(
                dist.precision.eval(), 1. / np.square(stddev))
            np.testing.assert_allclose(
                dist.log_precision.eval(), -2. * np.log(stddev))

    def test_sampling_with_log_std(self):
        with self.get_session(use_gpu=True):
            params = copy.copy(self.simple_params)
            params['logstd'] = np.log(params['stddev'])
            del params['stddev']

            x, prob, log_prob = self.get_samples_and_prob(**params)
            value_shape, batch_shape = \
                self.get_shapes_for_param(**self.simple_params)
            np.testing.assert_equal(x.shape, batch_shape + value_shape)
            self.assert_allclose(
                prob, self.prob(x, **self.simple_params))
            self.assert_allclose(
                log_prob, self.log_prob(x, **self.simple_params))

    def test_analytic_kld_with_log_std(self):
        with self.get_session(use_gpu=True):
            params = copy.copy(self.simple_params)
            params['logstd'] = np.log(params['stddev'])
            del params['stddev']

            kld_params = copy.copy(self.kld_simple_params)
            kld_params['logstd'] = np.log(kld_params['stddev'])
            del kld_params['stddev']

            self.assert_allclose(
                (self.dist_class(**params).
                 analytic_kld(self.dist_class(**kld_params))).eval(),
                self.analytic_kld(self.simple_params, self.kld_simple_params)
            )


if __name__ == '__main__':
    unittest.main()
