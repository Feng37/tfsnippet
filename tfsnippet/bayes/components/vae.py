# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.utils import VarScopeObject
from ..utils import StochasticObject

__all__ = ['VAE']


class DerivedVAE(StochasticObject):

    def __init__(self):
        pass

    @property
    def is_tight_log_lower_bound(self):
        return False

    def log_lower_bound(self, group_event_ndims=None, name=None):
        raise NotImplementedError()


class VAE(VarScopeObject):
    """Variational auto-encoder.

    This class provides a general implementation of the variational
    auto-encoder[1] and conditional variational auto-encoder[2].

    It can be used as a building block for more complex generative
    models, for example, a semi-supervised generative model[3] by
    feeding the label variable as the conditional input of the VAE.

    [1]	D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes”, vol.
        stat.ML. 21-Dec-2013.
    [2]	C. Doersch, “Tutorial on Variational Autoencoders.”, CoRR, 2016.
    [3]	D. P. Kingma, S. Mohamed, D. J. Rezende, and M. Welling,
        “Semi-supervised Learning with Deep Generative Models.”, NIPS, 2014.
    """

    def __init__(self,
                 x_dim,
                 z_dim,
                 x_distrib,
                 z_distrib,
                 model_net,
                 variational_net,
                 dtype=tf.float32,
                 name=None,
                 default_name=None):
        super(VAE, self).__init__(name=name, default_name=default_name)
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.x_distrib = x_distrib
        self.z_distrib = z_distrib
        self.model_net = model_net
        self.variational_net = variational_net
        self.dtype = tf.as_dtype(dtype)
