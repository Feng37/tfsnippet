# -*- coding: utf-8 -*-
import tensorflow as tf

__all__ = ['Distribution']


class Distribution(object):
    """Base class for various distributions."""

    def sample(self, sample_shape=()):
        """Sample from the distribution.

        Parameters
        ----------
        sample_shape : tuple[int | tf.Tensor]
            The shape of the samples.

        Returns
        -------
        tf.Tensor
            The samples as tensor.
        """
        raise NotImplementedError()

    def prob(self, x):
        """Compute the likelihood of `x` against the distribution.

        The extra dimensions at the front of `x` will be regarded as
        different samples, and the likelihood will be computed along
        these dimensions.

        Parameters
        ----------
        x : tf.Tensor
            The samples to be tested.

        Returns
        -------
        tf.Tensor
            The likelihood of `x`.
        """
        raise NotImplementedError()

    def log_prob(self, x):
        """Compute the log-probability of `x` against the distribution.

        Parameters
        ----------
        x : tf.Tensor
            The samples to be tested.

        Returns
        -------
        tf.Tensor
            The log-probability of `x`.
        """
        raise NotImplementedError()

    def analytic_kld(self, other):
        """Compute the KLD(self || other) using the analytic method.
        
        Parameters
        ----------
        other : Distribution
            The other distribution.
        
        Raises
        ------
        NotImplementedError
            If the KL-divergence of this distribution and `other` cannot be
            computed analytically.
    
        Returns
        -------
        tf.Tensor
            The KL-divergence.
        """
        raise NotImplementedError()
