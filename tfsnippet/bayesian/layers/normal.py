# -*- coding: utf-8 -*-
from .base import StochasticTensor

__all__ = ['Normal']


class Normal(StochasticTensor):
    """Stochastic tensor of normal distribution."""
