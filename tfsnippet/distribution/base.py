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
        tensor
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
        tensor
            The likelihood of `x`.
        """
        raise NotImplementedError()

    def log_prob(self, x):
        """Compute the log-likelihood of `x` against the distribution.

        Parameters
        ----------
        x : tf.Tensor
            The samples to be tested.

        Returns
        -------
        tensor
            The log-likelihood of `x`.
        """
        raise NotImplementedError()
