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
    group_event_ndims : int | tf.Tensor
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

    name, default_name : str
        Optional name or default name of this distribution.
    """

    def __init__(self, group_event_ndims=None, name=None, default_name=None):
        super(Distribution, self).__init__(name=name, default_name=default_name)
        self._group_event_ndims = group_event_ndims

    def __call__(self, n_samples=None, observed=None, group_event_ndims=None,
                 name=None):
        """Create a `StochasticTensor` with random samples or observations.

        Parameters
        ----------
        n_samples : int | tf.Tensor | None
            Generate this number of samples via `sample_n` method, unless
            `observed` is specified.

        observed : tf.Tensor | np.ndarray | float | int
            The observations.

        group_event_ndims : int | tf.Tensor
            If specify, override the default `group_event_ndims`.

        name : str
            Optional name of this operation.

        Returns
        -------
        tfsnippet.bayes.StochasticTensor
            The random samples or observations as tensor.
        """
        return self.sample_or_observe(n_samples=n_samples, observed=observed,
                                      group_event_ndims=group_event_ndims,
                                      name=name)

    @property
    def group_event_ndims(self):
        """Get the number of dimensions to be considered as events group.

        Returns
        -------
        int | tf.Tensor | None
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
        enum_observe
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

    def sample(self, shape=(), group_event_ndims=None, name=None):
        """Get random samples from the distribution.

        Parameters
        ----------
        shape : tuple[int | tf.Tensor] | tf.Tensor
            The shape of the samples, as a tuple of integers / 0-d tensors,
            or a 1-d tensor.

        group_event_ndims : int | tf.Tensor
            If specify, override the default `group_event_ndims`.

        name : str
            Optional name of this operation.

        Returns
        -------
        tfsnippet.bayes.StochasticTensor
            The random samples as `StochasticTensor`.
        """
        from ..stochastic import StochasticTensor
        with tf.name_scope(name, default_name='sample'):
            # derive the samples using ``n = prod(shape)``
            if is_deterministic_shape(shape):
                ret = self._sample_n(np.prod(shape, dtype=np.int32))
                static_sample_shape = tf.TensorShape(shape)
            else:
                if isinstance(shape, (tuple, list)):
                    static_sample_shape = tf.TensorShape(list(map(
                        lambda v: None if is_dynamic_tensor_like(v) else v,
                        shape
                    )))
                    shape = tf.stack(shape, axis=0)
                else:
                    static_sample_shape = tf.TensorShape(None)
                ret = self._sample_n(tf.reduce_prod(shape))

            # reshape the samples
            tail_shape = ret.get_shape()[1:]
            with tf.name_scope('reshape'):
                dynamic_shape = tf.concat(
                    [shape, tf.shape(ret)[1:]], axis=0)
                ret = tf.reshape(ret, dynamic_shape)

            # special fix: when `sample_shape` is a tensor, it would cause
            #   `static_sample_shape` to carry no information.  We thus
            #   re-capture the sample shape by query at the reshaped samples.
            if tail_shape.ndims is not None and \
                    ret.get_shape().ndims is not None and \
                    static_sample_shape.ndims is None:
                tail_ndims = tail_shape.ndims
                if tail_ndims == 0:
                    static_sample_shape = ret.get_shape()
                else:
                    static_sample_shape = ret.get_shape()[: -tail_ndims]

            # fix the static shape of samples
            ret.set_shape(static_sample_shape.concatenate(tail_shape))

            if group_event_ndims is None:
                group_event_ndims = self.group_event_ndims
            return StochasticTensor(
                self,
                samples=ret,
                group_event_ndims=group_event_ndims
            )

    def sample_n(self, n=None, group_event_ndims=None, name=None):
        """Get random samples from the distribution.

        The different between this method and `sample` is that, this method
        expects a 0-d tensor as the count of samples, but `sample` expects
        a 1-d tensor as the shape of samples.

        Parameters
        ----------
        n : int | tf.Tensor | None
            The number of samples to take.

            If not specified, only one sample will be taken correspond
            to each set of parameter, while the shape of samples will
            equal to ``batch_shape + value_shape``.  Otherwise the
            specified number of samples will be taken, while the shape
            will equal to ``(n,) + batch_shape + value_shape``.

        group_event_ndims : int | tf.Tensor
            If specify, override the default `group_event_ndims`.

        name : str
            Optional name of this operation.

        Returns
        -------
        tfsnippet.bayes.StochasticTensor
            The random samples as tensor.
        """
        from ..stochastic import StochasticTensor
        with tf.name_scope(name, default_name='sample_n'):
            if n is None:
                samples = tf.squeeze(self._sample_n(1), 0)
            else:
                samples = self._sample_n(n)

            if group_event_ndims is None:
                group_event_ndims = self.group_event_ndims
            return StochasticTensor(
                self,
                samples=samples,
                group_event_ndims=group_event_ndims
            )

    def observe(self, observed, group_event_ndims=None):
        """Create a `StochasticTensor` with specified observations.

        Parameters
        ----------
        observed : tf.Tensor | np.ndarray | float | int
            The observations.

        group_event_ndims : int | tf.Tensor
            If specify, override the default `group_event_ndims`.

        Returns
        -------
        tfsnippet.bayes.StochasticTensor
            The stochastic tensor.
        """
        from ..stochastic import StochasticTensor
        if group_event_ndims is None:
            group_event_ndims = self.group_event_ndims
        return StochasticTensor(
            self,
            observed=observed,
            group_event_ndims=group_event_ndims
        )

    def sample_or_observe(self, n_samples=None, observed=None,
                          group_event_ndims=None, name=None):
        """Create a `StochasticTensor` with random samples or observations.

        Parameters
        ----------
        n_samples : int | tf.Tensor | None
            Generate this number of samples via `sample_n` method, unless
            `observed` is specified.

        observed : tf.Tensor | np.ndarray | float | int
            The observations.

        group_event_ndims : int | tf.Tensor
            If specify, override the default `group_event_ndims`.

        name : str
            Optional name of this operation.

        Returns
        -------
        tfsnippet.bayes.StochasticTensor
            The random samples or observations as tensor.
        """
        if observed is None:
            return self.sample_n(
                n_samples, group_event_ndims=group_event_ndims, name=name)
        else:
            return self.observe(
                observed, group_event_ndims=group_event_ndims)

    def _enum_values(self):
        raise RuntimeError('%s is not enumerable.' %
                           self.__class__.__name__)

    def enum_sample(self, group_event_ndims=None, name=None):
        """Enumerate possible values as samples.

        The returned `StochasticTensor` should in the shape of
        ``(enum_value_count,) + batch_shape + value_shape``, where
        `enum_value_count` is the count of possible values from this
        distribution.

        Parameters
        ----------
        group_event_ndims : int | tf.Tensor
            If specify, override the default `group_event_ndims`.

        name : str
            Optional name of this operation.

        Returns
        -------
        tfsnippet.bayes.StochasticTensor
            The enumerated values as samples.

        Raises
        ------
        RuntimeError
            If the distribution is not enumerable.
        """
        from ..stochastic import StochasticTensor
        with tf.name_scope(name, default_name='enum_observe'):
            if group_event_ndims is None:
                group_event_ndims = self.group_event_ndims
            return StochasticTensor(
                self,
                samples=self._enum_values(),
                group_event_ndims=group_event_ndims
            )

    def enum_observe(self, group_event_ndims=None, name=None):
        """Enumerate possible values as observations.

        The returned `StochasticTensor` should in the shape of
        ``(enum_value_count,) + batch_shape + value_shape``, where
        `enum_value_count` is the count of possible values from this
        distribution.

        Parameters
        ----------
        group_event_ndims : int | tf.Tensor
            If specify, override the default `group_event_ndims`.

        name : str
            Optional name of this operation.

        Returns
        -------
        tfsnippet.bayes.StochasticTensor
            The enumerated values as observations.

        Raises
        ------
        RuntimeError
            If the distribution is not enumerable.
        """
        from ..stochastic import StochasticTensor
        with tf.name_scope(name, default_name='enum_observe'):
            if group_event_ndims is None:
                group_event_ndims = self.group_event_ndims
            return StochasticTensor(
                self,
                observed=self._enum_values(),
                group_event_ndims=group_event_ndims
            )

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

        group_event_ndims : int | tf.Tensor
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

            # reduce the dimensions of group event
            def f(ndims):
                return tf.reduce_sum(log_prob, axis=tf.range(-ndims, 0))

            if group_event_ndims is not None:
                if is_integer(group_event_ndims):
                    if group_event_ndims > 0:
                        log_prob = f(group_event_ndims)
                else:
                    group_event_ndims = tf.convert_to_tensor(group_event_ndims,
                                                             dtype=tf.int32)
                    return tf.cond(
                        group_event_ndims > 0,
                        lambda: f(group_event_ndims),
                        lambda: log_prob
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

        group_event_ndims : int | tf.Tensor
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
