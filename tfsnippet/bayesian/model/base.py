# -*- coding: utf-8 -*-
from tfsnippet.utils import VarScopeObject

__all__ = ['StochasticTensor']


class StochasticTensor(VarScopeObject):
    """Tensor-like object that represents a random variable in graph.

    Parameters
    ----------
    distrib : Distribution
        The distribution of this stochastic tensor.
    """

    def __init__(self, distrib):
        self.distrib = distrib
