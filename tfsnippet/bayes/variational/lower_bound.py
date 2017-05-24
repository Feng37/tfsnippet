# -*- coding: utf-8 -*-

"""Estimators to compute the variational lower bound.

The variational lower bound is a certain kind of estimators using variational
methods.  It is mainly used to compute the approximated log-likelihood of
a Bayesian network, given the observed variables, as well as the prior and
posterior of latent variables.
"""

import tensorflow as tf

from ..utils import gather_log_lower_bound

__all__ = [
    'sgvb'
]


def sgvb(model, variational, latent_axis=None, name=None):
    """SGVB estimator for the variational lower bound.

    The SGVB estimator[1], if given observed variable `x` and latent variable
    `z`, where the posterior of `x` is represented by `p(x|z)`, the prior of
    `z` is represented by `p(z)` and the approximated posterior of `z` is
    represented by `q(z|x)`, then it computes the variational lower bound by:

        .. math::
            \\log p(x)
                &\\geq \\log p(x) - \\text{KL} \\left(q(z|x) \\| p(z|x) \\right)
                      \\\\
                &= \\mathop{\\mathbb{E}_{q(z|x)}} \\left[
                        \\log p(x) + \\log p(z|x) - \\log q(z|x) \\right] \\\\
                &= \\mathop{\\mathbb{E}_{q(z|x)}} \\left[
                        \\log p(x,z) - \\log q(z|x) \\right]

    Note that SGVB can only be applied on continuous variables which can be
    re-parameterized.

    [1]	D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” vol.
        stat.ML. 21-Dec-2013.

    Parameters
    ----------
    model : collections.Iterable[StochasticObject | tf.Tensor]
        List of stochastic objects or log-probability lower-bound tensors
        from both observed and latent variables in the generative model.

    variational : collections.Iterable[StochasticObject | tf.Tensor]
        List of stochastic tensors or log-probability lower-bound tensors
        from approximated latent variables.

    latent_axis : int | list[int] | tf.Tensor
        The dimension(s) of samples, which should be averaged in order
        to compute the expectation of variational lower bound.

        If not specified, then the expectation will not be computed.

    name : str
        Optional name of this operation.

    Returns
    -------
    tf.Tensor
        The computed variational lower bound.
    """
    with tf.name_scope(name, default_name='sgvb'):
        model_log_prob = sum(gather_log_lower_bound(model))
        variational_entropy = -sum(gather_log_lower_bound(variational))
        lower_bound = model_log_prob + variational_entropy
        if latent_axis is not None:
            lower_bound = tf.reduce_mean(lower_bound, axis=latent_axis)
        return lower_bound
