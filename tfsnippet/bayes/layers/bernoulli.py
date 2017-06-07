# -*- coding: utf-8 -*-
from .base import StochasticLayer
from ..distributions import Bernoulli

__all__ = ['BernoulliLayer']


class BernoulliLayer(StochasticLayer):
    """Stochastic layer for Bernoulli distribution.

    A `NormalLayer` expects `logits` or `probs` in layer inputs,
    for example:

        bernoulli = BernoulliLayer(n_samples=100, group_event_ndims=1)
        x = bernoulli({'logits': x_logits})
        y = bernoulli(probs=x_probs)
    """

    def _call(self, logits=None, probs=None, n_samples=None,
              observed=None, group_event_ndims=None):
        bernoulli = Bernoulli(logits=logits, probs=probs,
                              group_event_ndims=group_event_ndims)
        return bernoulli.sample_or_observe(
            n_samples=n_samples, observed=observed
        )
