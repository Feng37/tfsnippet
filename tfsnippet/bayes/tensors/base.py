# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.distributions import Distribution
from tfsnippet.utils import (VarScopeObject,
                             is_integer,
                             reopen_variable_scope,
                             TensorArithmeticMixin,
                             instance_reuse)

__all__ = ['StochasticTensor']


class StochasticTensor(VarScopeObject, TensorArithmeticMixin):
    """Tensor-like object that represents a stochastic variable.

    A `StochasticTensor` should be constructed from a `distribution`,
    and represents a stochastic variable in the model.  Although it
    stands for a random variable, it is actually a standard tensor
    in TensorFlow graph, where the randomness is achieved by random
    sampling.  That is to say, every `StochasticTensor` is equivalent
    to a set of random samples from the given `distribution`.
    The samples are stored so that every stochastic tensor should
    always represent the same set of samples.

    Number of samples are controlled by `sample_num` argument.
    Sometimes a stochastic tensor need to be fixed using a set of
    observations.  This could be achieved by `observed` argument.

    Parameters
    ----------
    distribution : Distribution | () -> Distribution
        The distribution of this stochastic tensor, or a factory that
        constructs the distribution.

    sample_num : int | tf.Tensor | tf.Variable
        Optional number of samples to take. (default None)

        If specified, even if 1, the sample(s) will occupy a new dimension
        before `batch_shape + value_shape` of `distrib`.
        Otherwise only one sample will be taken and it will not occupy
        a new dimension.

    enum_as_samples : bool
        Whether or not to take enumeration "samples" rather than
        random samples? (default False)

        This argument cannot be True when `sample_num` is specified.
        If the distribution is not enumerate but this argument is
        set to True, a `TypeError` will be raised.

    observed : tf.Tensor
        If specified, random sampling will not take place.
        Instead this tensor will be regarded as sampled tensor.
        (default None)

        Note that if `sample_num` is specified, the shape of this
        tensor must be `[?] + batch_shape + value_shape`.
        And if `sample_num` is not specified, its shape must be
        `batch_shape + value_shape`.

    validate_observed_shape : bool
        Whether or not to validate the shape of `observed`? (default False)

        If set to True, the shape of `observed` must match that returned
        by `get_shape()`, given `sample_num` argument.

    group_event_ndims : int
        If specify, override the default `group_event_ndims` of `distribution`.
        This argument can further be overrided by `group_event_ndims` argument
        of `prob` and `log_prob` method.

    name, default_name : str
        Optional name or default name of this stochastic tensor.
    """

    def __init__(self, distribution, sample_num=None, enum_as_samples=False,
                 observed=None, validate_observed_shape=False,
                 group_event_ndims=None, name=None, default_name=None):
        if sample_num is not None and enum_as_samples:
            raise ValueError('`enum_as_samples` cannot be True when '
                             '`sample_num` is specified.')

        super(StochasticTensor, self).__init__(
            name=name,
            default_name=default_name
        )

        with reopen_variable_scope(self.variable_scope):
            if not isinstance(distribution, Distribution) and \
                    callable(distribution):
                # we support the factory of distributions, so that
                # the scope of distribution can be contained in the
                # scope of stochastic tensor.
                distribution = distribution()

        if not isinstance(distribution, Distribution):
            raise TypeError(
                'Expected a Distribution but got %r.' % (distribution,))

        if enum_as_samples and not distribution.is_enumerable:
            raise TypeError(
                '`enum_as_samples` is True but %r is not enumerable.' %
                (distribution,)
            )

        with reopen_variable_scope(self.variable_scope):
            with tf.name_scope('init'):
                if group_event_ndims is not None:
                    if not is_integer(group_event_ndims) or \
                            group_event_ndims < 0:
                        raise TypeError(
                            '`group_event_ndims` must be a non-negative '
                            'integer constant.'
                        )
                else:
                    group_event_ndims = distribution.group_event_ndims

                if sample_num is not None:
                    if is_integer(sample_num):
                        if sample_num < 1:
                            raise ValueError('`sample_num` must be at least 1.')
                        sample_shape = (sample_num,)
                        static_sample_shape = (sample_num,)
                    else:
                        sample_num = tf.convert_to_tensor(sample_num)
                        sample_num_shape = sample_num.get_shape()
                        if not sample_num.dtype.is_integer or (
                                        sample_num_shape.ndims is not None and
                                        sample_num_shape.ndims != 0):
                            raise ValueError('`sample_num` must be an integer '
                                             'scalar.')
                        sample_shape = tf.stack([sample_num])
                        static_sample_shape = (None,)
                else:
                    sample_shape = ()
                    static_sample_shape = ()

                static_shape = (
                    tf.TensorShape(list(static_sample_shape)).
                        concatenate(distribution.static_batch_shape).
                        concatenate(distribution.static_value_shape)
                )
                if observed is not None:
                    observed = tf.convert_to_tensor(
                        observed,
                        dtype=distribution.dtype
                    )
                    observed_shape = observed.get_shape()
                    if validate_observed_shape and \
                            not static_shape.is_compatible_with(observed_shape):
                        raise ValueError('The shape of observed is %r, which '
                                         'is not compatible with the shape %r '
                                         'of the StochasticTensor.' %
                                         (observed_shape, static_shape))
                    static_shape = observed_shape

                self._distrib = distribution
                self._sample_num = sample_num
                self._enum_as_samples = enum_as_samples
                self._group_event_ndims = group_event_ndims
                self._sample_shape = sample_shape
                self._static_shape = static_shape
                self._observed_tensor = observed
                self._computed_tensor = None

    def __repr__(self):
        return 'StochasticTensor(%r)' % (self.computed_tensor,)

    def __hash__(self):
        # Necessary to support Python's collection membership operators
        return id(self)

    def __eq__(self, other):
        # Necessary to support Python's collection membership operators
        return id(self) == id(other)

    def get_shape(self):
        """Get the static shape of this stochastic tensor.

        Returns
        -------
        tf.TensorShape
            The static shape of this stochastic tensor.
        """
        return self._static_shape

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
    def distribution(self):
        """Get the distribution."""
        return self._distrib

    @property
    def dtype(self):
        """Get the data type of this stochastic tensor."""
        return self._distrib.dtype

    @property
    def sample_num(self):
        """Get the sample number."""
        return self._sample_num

    @property
    def is_observed(self):
        """Whether or not this stochastic tensor is observed?"""
        return self._observed_tensor is not None

    @property
    def observed_tensor(self):
        """Get the observed tensor, if specified.

        Returns
        -------
        tf.Tensor
            The observed tensor.
        """
        return self._observed_tensor

    def _sample_tensor(self):
        with reopen_variable_scope(self.variable_scope):
            if self._enum_as_samples:
                return self._distrib.enum_sample()
            return self._distrib.sample(self._sample_shape)

    @property
    def computed_tensor(self):
        """Get the observed or sampled tensor.

        Returns
        -------
        tf.Tensor
            The observed or sampled tensor.
        """
        if self._computed_tensor is None:
            if self._observed_tensor is not None:
                self._computed_tensor = self._observed_tensor
            else:
                self._computed_tensor = self._sample_tensor()
        return self._computed_tensor

    def prob(self, group_event_ndims=None, name=None):
        """Compute the likelihood of this stochastic tensor.

        Parameters
        ----------
        group_event_ndims : int
            If specified, will override the `group_event_ndims` configured
            in both this stochastic tensor and the distribution.

        name : str
            Optional name of this operation.

        Returns
        -------
        tf.Tensor
            The likelihood of this stochastic tensor.
        """
        if group_event_ndims is None:
            group_event_ndims = self._group_event_ndims
        return self._distrib.prob(
            self.computed_tensor, group_event_ndims=group_event_ndims,
            name=name
        )

    def log_prob(self, group_event_ndims=None, name=None):
        """Compute the log-likelihood of this stochastic tensor.

        Parameters
        ----------
        group_event_ndims : int
            If specified, will override the `group_event_ndims` configured
            in both this stochastic tensor and the distribution.

        name : str
            Optional name of this operation.

        Returns
        -------
        tf.Tensor
            The log-likelihood of this stochastic tensor.
        """
        if group_event_ndims is None:
            group_event_ndims = self._group_event_ndims
        return self._distrib.log_prob(
            self.computed_tensor, group_event_ndims=group_event_ndims,
            name=name
        )


def _stochastic_to_tensor(value, dtype=None, name=None, as_ref=False):
    if dtype and not dtype.is_compatible_with(value.dtype):
        raise ValueError('Incompatible type conversion requested to type '
                         '%s for variable of type %s' %
                         (dtype.name, value.dtype.name))
    if as_ref:
        raise ValueError('%r: Ref type not supported.' % value)
    return value.computed_tensor


tf.register_tensor_conversion_function(StochasticTensor, _stochastic_to_tensor)
