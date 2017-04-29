# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.utils import (VarScopeObject, is_integer, instance_reuse)

__all__ = ['Distribution']


def _reduce_group_events(x, ):
    pass


class Distribution(VarScopeObject):
    """Base class for various distributions.
    
    
    
    Parameters
    ----------
    group_event_ndims : int
        If specify, then this number of dimensions at the end of `batch_shape`
        would be considered as a group of single events.
    """

    def __init__(self, group_event_ndims=None, name=None, default_name=None):
        if group_event_ndims is not None:
            if not is_integer(group_event_ndims) or group_event_ndims < 0:
                raise TypeError(
                    '`group_event_ndims` must be a non-negative integer '
                    'constant.'
                )

        super(Distribution, self).__init__(name=name, default_name=default_name)
        self._group_event_ndims = group_event_ndims

    @property
    def group_event_ndims(self):
        """"""
        return self._group_event_ndims

    @property
    def dynamic_batch_shape(self):
        """Get the dynamic batch shape of this distribution."""
        raise NotImplementedError()

    @property
    def static_batch_shape(self):
        """Get the static batch shape of this distribution."""
        raise NotImplementedError()

    @property
    def mean(self):
        """Get the mean of this distribution."""
        raise NotImplementedError()

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

    def _prob(self, x):
        # `@instance_reuse` decorator should not be applied to this method.
        raise NotImplementedError()

    @instance_reuse
    def prob(self, x, group_event_ndims=None):
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
        return self._prob(x)

    def _log_prob(self, x):
        # `@instance_reuse` decorator should not be applied to this method.
        raise NotImplementedError()

    @instance_reuse
    def log_prob(self, x, group_event_ndims=None):
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
        return self._log_prob(x)

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
