# -*- coding: utf-8 -*-
from tfsnippet.components import Component
from ..utils import StochasticObject

__all__ = ['VAE']


class VAE(Component, StochasticObject):
    """Variational auto-encoder.

    This class provides a general implementation of the variational
    auto-encoder[1] and conditional variational auto-encoder[2].

    [1]	D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” vol.
        stat.ML. 21-Dec-2013.
    [2]	C. Doersch, “Tutorial on Variational Autoencoders.,” CoRR, 2016.
    """

    def __init__(self,
                 z_dim,
                 z_distrib,
                 x_distrib,
                 variational_net,
                 generative_net,
                 z_sample_num=None,
                 x_sample_num=None,
                 x=None,
                 y=None,
                 z=None,
                 name=None,
                 default_name=None):
        super(VAE, self).__init__(name=name, default_name=default_name)

    @property
    def is_log_prob_normalized(self):
        return False

    def log_prob(self, group_event_ndims=None, name=None):
        raise NotImplementedError()
