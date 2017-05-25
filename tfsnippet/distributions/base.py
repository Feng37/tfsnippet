# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tfsnippet.utils import (VarScopeObject, is_integer, is_deterministic_shape,
                             is_dynamic_tensor_like)

__all__ = ['Distribution']


class Distribution(VarScopeObject):
    """Base class for various distributions.

    A distribution object should have unchangeable parameters, whose values
    should be specified on construction.

    Besides, a batch of parameters may be specified instead of just one set
    of parameter, so that a batch of random variables can be sampled at the
    same time.  In order to achieve this, the shapes of each parameter
    should be composed of two parts: the `batch_shape` and the corresponding
    `value_shape` of the parameter.

    Any parameter should contain at least the same dimension as `value_shape`,
    while it can contain more dimensions than `batch_shape`.  Besides, if any
    one of the parameters have more dimensions than another, these two
    parameters must be broadcastable.

    When the `batch_shape` of specified parameters have more than 1 dimension,
    it might contain groups of events, whose probabilities should be accounted
    together.  Such requirement can be achieved by `group_event_ndims` argument
    of the constructor, as well as the `prob(x)` and `log_prob(x)` method.

    Parameters
    ----------
    group_event_ndims : int
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

    name, default_name : str
        Optional name or default name of this distribution.
    """

    def __init__(self, group_event_ndims=None, name=None, default_name=None):
        if group_event_ndims is not None:
            if not is_integer(group_event_ndims) or group_event_ndims < 0:
                raise TypeError(
                    '`group_event_ndims` must be a non-negative integer '
                    'constant.'
                )

        super(Distribution, self).__init__(name=name, default_name=default_name)
        # TODO: support dynamic `group_event_ndims` if necessary
        self._group_event_ndims = group_event_ndims

    @property
    def group_event_ndims(self):
        """Get the number of dimensions to be considered as events group.

        Returns
        -------
        int | None
            The number of dimensions.  If `group_event_ndims` is not
            specified in the constructor, will return None.
        """
        return self._group_event_ndims

    @property
    def dtype(self):
        """Get the data type of samples.

        Returns
        -------
        tf.DType
            Data type of the samples.
        """
        raise NotImplementedError()

    @property
    def param_dtype(self):
        """Get the data type of parameter(s).

        Returns
        -------
        tf.DType
            Data type of the parameter(s).
        """
        raise NotImplementedError()

    @property
    def is_continuous(self):
        """Whether or not the distribution is continuous?

        Returns
        -------
        bool
            A boolean indicating whether the distribution is continuous.
        """
        raise NotImplementedError()

    @property
    def is_reparameterized(self):
        """Whether or not the distribution is re-parameterized?

        Returns
        -------
        bool
            A boolean indicating whether the distribution is re-parameterized.
        """
        raise NotImplementedError()

    @property
    def is_enumerable(self):
        """Whether or not the distribution is enumerable?

        A distribution with a finite value range is enumerable.
        Enumerable distributions could derive a special set of "samples",
        such that the probability of every possible value against each
        individual set of parameters could be computed.

        Returns
        -------
        bool
            A boolean indicating whether the distribution is enumerable.

        See Also
        --------
        enum_sample
        """
        raise NotImplementedError()

    @property
    def enum_value_count(self):
        """Get the count of possible values from this distribution.

        Returns
        -------
        int | tf.Tensor | None
            Static or dynamic count of possible values.
            If the distribution is not enumerable, it should return None.
        """
        raise NotImplementedError()

    @property
    def dynamic_batch_shape(self):
        """Get the dynamic batch shape of this distribution.

        Returns
        -------
        tf.Tensor
            The a tensor as the batch shape.
        """
        raise NotImplementedError()

    @property
    def static_batch_shape(self):
        """Get the static batch shape of this distribution.

        Returns
        -------
        tf.TensorShape
            The tensor shape object corresponding to `dynamic_batch_shape`.
        """
        raise NotImplementedError()

    @property
    def dynamic_value_shape(self):
        """Get the dynamic value shape of this distribution.

        Returns
        -------
        tf.Tensor
            A tensor as the dynamic value shape.
        """
        raise NotImplementedError()

    @property
    def static_value_shape(self):
        """Get the static value shape of this distribution.

        Returns
        -------
        tf.TensorShape
            The tensor shape object corresponding to `dynamic_value_shape`.
        """
        raise NotImplementedError()

    def _sample_n(self, n):
        # `@instance_reuse` decorator should not be applied to this method.
        raise NotImplementedError()

    def sample(self, sample_shape=(), name=None):
        """Get random samples from the distribution.

        Parameters
        ----------
        sample_shape : tuple[int | tf.Tensor] | tf.Tensor
            The shape of the samples, as a tuple of integers / 0-d tensors,
            or a 1-d tensor.

        name : str
            Optional name of this operation.

        Returns
        -------
        tf.Tensor
            The random samples as tensor.
        """
        with tf.name_scope(name, default_name='sample'):
            # derive the samples using ``n = prod(sample_shape)``
            if is_deterministic_shape(sample_shape):
                ret = self._sample_n(np.prod(sample_shape, dtype=np.int32))
                static_sample_shape = tf.TensorShape(sample_shape)
            else:
                if isinstance(sample_shape, (tuple, list)):
                    static_sample_shape = tf.TensorShape(list(filter(
                        lambda v: None if is_dynamic_tensor_like(v) else v,
                        sample_shape
                    )))
                    sample_shape = tf.stack(sample_shape, axis=0)
                else:
                    static_sample_shape = tf.TensorShape(None)
                ret = self._sample_n(tf.reduce_prod(sample_shape))

            # reshape the samples
            static_batch_shape = ret.get_shape()[1:]
            with tf.name_scope('reshape'):
                dynamic_shape = tf.concat(
                    [sample_shape, tf.shape(ret)[1:]], axis=0)
                ret = tf.reshape(ret, dynamic_shape)

            # special fix: when `sample_shape` is a tensor, it would cause
            #   `static_sample_shape` to carry no information.  We thus
            #   re-capture the sample shape by query at the reshaped samples.
            if static_batch_shape.ndims is not None and \
                    ret.get_shape().ndims is not None and \
                    static_sample_shape.ndims is None:
                tail_ndims = static_batch_shape.ndims
                if tail_ndims == 0:
                    static_sample_shape = ret.get_shape()
                else:
                    static_sample_shape = ret.get_shape()[: -tail_ndims]

            # fix the static shape of samples
            ret.set_shape(static_sample_shape.concatenate(static_batch_shape))
            return ret

    def _enum_sample(self):
        raise RuntimeError('%s distribution is not enumerable.' %
                           self.__class__.__name__)

    def enum_sample(self, name=None):
        """Get enumeration "samples" from the distribution.

        The returned samples should be of shape ``(enum_value_count,) +
        batch_shape + value_shape``, where `enum_value_count` is the count
        of possible values from this distribution.

        Parameters
        ----------
        name : str
            Optional name of this operation.

        Returns
        -------
        tf.Tensor
            The enumeration "samples" as tensor.

        Raises
        ------
        RuntimeError
            If the distribution is not enumerable.
        """
        with tf.name_scope(name, default_name='enum_sample'):
            return self._enum_sample()

    def _log_prob(self, x):
        # `@instance_reuse` decorator should not be applied to this method.
        raise NotImplementedError()

    def log_prob(self, x, group_event_ndims=None, name=None):
        """Compute the log-probability of `x` against the distribution.

        If `group_event_ndims` is configured, then the likelihoods of
        one group of events will be summed together.

        Parameters
        ----------
        x : tf.Tensor
            The samples to be tested.

        group_event_ndims : int
            If specified, will override the attribute `group_event_ndims`
            of this distribution object.

        name : str
            Optional name of this operation.

        Returns
        -------
        tf.Tensor
            The log-probability of `x`.
        """
        with tf.name_scope(name, default_name='log_prob'):
            # determine the number of group event dimensions
            if group_event_ndims is None:
                group_event_ndims = self.group_event_ndims

            # compute the log-likelihood
            log_prob = self._log_prob(x)
            log_prob_shape = log_prob.get_shape()
            try:
                tf.broadcast_static_shape(log_prob_shape,
                                          self.static_batch_shape)
            except ValueError:
                raise RuntimeError(
                    'The shape of computed log-prob does not match '
                    '`batch_shape`, which could be a bug in the distribution '
                    'implementation. (%r vs %r)' %
                    (log_prob_shape, self.static_batch_shape)
                )

            if group_event_ndims:
                log_prob = tf.reduce_sum(
                    log_prob,
                    axis=tf.range(-group_event_ndims, 0)
                )
            return log_prob

    def prob(self, x, group_event_ndims=None, name=None):
        """Compute the likelihood of `x` against the distribution.

        The extra dimensions at the front of `x` will be regarded as
        different samples, and the likelihood will be computed along
        these dimensions.

        If `group_event_ndims` is configured, then the likelihoods of
        one group of events will be multiplied together.

        Parameters
        ----------
        x : tf.Tensor
            The samples to be tested.

        group_event_ndims : int
            If specified, will override the attribute `group_event_ndims`
            of this distribution object.

        name : str
            Optional name of this operation.

        Returns
        -------
        tf.Tensor
            The likelihood of `x`.
        """
        with tf.name_scope(name, default_name='prob'):
            return tf.exp(self.log_prob(x, group_event_ndims=group_event_ndims))

    def _analytic_kld(self, other):
        # `@instance_reuse` decorator should not be applied to this method.
        raise NotImplementedError()

    def analytic_kld(self, other, name=None):
        """Compute the KLD(self || other) using the analytic method.

        Parameters
        ----------
        other : Distribution
            The other distribution.

        name : str
            Optional name of this operation.

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
        with tf.name_scope(name, default_name='analytic_kld'):
            ret = self._analytic_kld(other)
            if ret is None:
                raise NotImplementedError()
            return ret
