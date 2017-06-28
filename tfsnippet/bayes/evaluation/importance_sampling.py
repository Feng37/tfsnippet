# -*- coding: utf-8 -*-
import tensorflow as tf

__all__ = ['importance_sampling']


def importance_sampling(arr, log_p, log_q, latent_axis=None,
                        stop_gradient=False, name=None):
    """Importance sampling for Monte Carlo integration.

    The importance sampling method is usually used to compute the
    expectation of `f(x)` over distribution `p(x)`, however, using an
    alternative distribution `q(x)` to sample `x`, instead of using
    the true distribution `p(x)`.  It computes the integration by:

        .. math::
            \\begin{aligned}
                \\int p(x) f(x) \\,\\mathrm{d}x
                    &= \\int q(x) \\frac{p(x)}{q(x)} f(x) \\,\\mathrm{d}x \\\\
                    &= \\int q(x) \\exp\\left(\\log p(x) - \\log q(x)\\right)
                       f(x) \\,\\mathrm{d}x
            \\end{aligned}

    Parameters
    ----------
    arr : tf.Tensor
        The array of `f(x)` to be integrated by importance sampling.

    log_p : tf.Tensor
        The log-probability :math:`\\log p(x)`.

    log_q : tf.Tensor
        The log-probability :math:`\\log q(x)`.

    stop_gradient : bool
        Whether or not to stop the gradient propagation along importance
        sampling factor :math:`\\exp \\left(\\log p(x) - \\log q(x)\\right)`?
        (default False)

    latent_axis : int | list[int] | tf.Tensor
        The dimension(s) of samples, which should be averaged in order
        to compute the expectation of variational lower bound.

        If not specified, then the expectation will not be computed.

    name : str
        Optional name of this operation.

    Returns
    -------
    tf.Tensor
        The computed integration (before or after the samples being
        averaged).
    """
    with tf.name_scope(name, default_name='importance_sampling'):
        factor = tf.exp(log_p - log_q, name='importance_factor')
        if stop_gradient:
            factor = tf.stop_gradient(factor)
        integrated = arr * factor
        if latent_axis is not None:
            integrated = tf.reduce_mean(integrated, axis=latent_axis)
        return integrated
