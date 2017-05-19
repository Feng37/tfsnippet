# -*- coding: utf-8 -*-
import unittest

import six
import numpy as np
from scipy.special import gammaln, digamma

from tfsnippet.distributions import Gamma
from tests.distributions._helper import (UnivariateDistributionTestMixin,
                                         AnalyticKldTestMixin)
from tests.helper import TestCase


class GammaTestCase(TestCase,
                    UnivariateDistributionTestMixin,
                    AnalyticKldTestMixin):
    dist_class = Gamma
    simple_params = {
        'alpha': np.asarray([0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0, 2.0],
                            dtype=np.float32),
        'beta': np.asarray([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                           dtype=np.float32)
    }
    extended_dimensional_params = {
        k: v + np.asarray([[0.0], [0.1]], dtype=np.float32)
        for k, v in six.iteritems(simple_params)
    }
    kld_simple_params = {
        'alpha': np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                            dtype=np.float32),
        'beta': np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                           dtype=np.float32)
    }
    is_continuous = True
    is_reparameterized = False

    def get_shapes_for_param(self, **params):
        x = params['alpha'] + params['beta']
        return (), x.shape

    def log_prob(self, x, group_event_ndims=None, **params):
        alpha, beta = params['alpha'], params['beta']
        log_prob = (alpha * np.log(beta) + (alpha - 1) * np.log(x) -
                    beta * x - gammaln(alpha))
        if group_event_ndims:
            grouped_shape = log_prob.shape[: -group_event_ndims] + (-1,)
            log_prob = np.sum(log_prob.reshape(grouped_shape), axis=-1)
        return log_prob

    def get_mean_stddev(self, **params):
        alpha, beta = params['alpha'], params['beta']
        return alpha / beta, alpha / (beta ** 2)

    def analytic_kld(self, params1, params2):
        alpha_1, beta_1 = params1['alpha'], params1['beta']
        alpha_2, beta_2 = params2['alpha'], params2['beta']
        return (
            (alpha_1 - alpha_2) * digamma(alpha_1) -
            gammaln(alpha_1) + gammaln(alpha_2) +
            alpha_2 * (np.log(beta_1) - np.log(beta_2)) +
            alpha_1 * (beta_2 / beta_1 - 1.)
        )

    # test cases for Gamma distribution
    def test_construction_error(self):
        with self.test_session():
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

    def test_other_properties(self):
        with self.test_session():
            dist = Gamma(**self.simple_params)
            self.assert_allclose(dist.alpha.eval(), self.simple_params['alpha'])
            self.assert_allclose(dist.beta.eval(), self.simple_params['beta'])


if __name__ == '__main__':
    unittest.main()
