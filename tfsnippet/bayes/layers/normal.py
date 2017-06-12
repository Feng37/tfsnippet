# -*- coding: utf-8 -*-
from .base import StochasticLayer
from ..distributions import Normal

__all__ = ['NormalLayer']


class NormalLayer(StochasticLayer):
    """Stochastic layer for Normal distribution.

    A `NormalLayer` expects `mean` and `stddev`/`logstd` in layer inputs,
    for example:

        normal = NormalLayer(n_samples=100, group_event_ndims=1)
        x = normal({'mean': x_mean, 'stddev': x_stddev})
        y = normal(mean=y_mean, logstd=y_logstd)
    """

    def _call(self, mean, stddev=None, logstd=None, n_samples=None,
              observed=None, group_event_ndims=None,
              check_numerics=False, validate_shape=False):
        normal = Normal(mean=mean, stddev=stddev, logstd=logstd,
                        group_event_ndims=group_event_ndims,
                        check_numerics=check_numerics)
        return normal.sample_or_observe(
            n_samples=n_samples, observed=observed,
            validate_shape=validate_shape
        )
