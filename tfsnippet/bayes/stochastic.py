# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.utils import TensorArithmeticMixin
from .distributions import Distribution

__all__ = [
    'StochasticObject',
    'StochasticTensor',
]


class StochasticObject(object):
    """Base interface for stochastic objects.

    A stochastic object should be any object in a TensorFlow model,
    which has a log-probability lower-bound.
    """

    def log_lower_bound(self, group_event_ndims=None, name=None):
        """Compute the log-probability lower-bound.

        Parameters
        ----------
        group_event_ndims : int
            If specify, this number of dimensions at the end of `batch_shape`
            would be considered as a group of events, whose log-probability
            lower-bounds are summed together. (default None)

        name : str
            Optional name of this operation.

        Returns
        -------
        tf.Tensor
            The log-probability lower-bound.
        """
        raise NotImplementedError()


class StochasticTensor(StochasticObject, TensorArithmeticMixin):
    """Tensor-like object that represents a stochastic variable.

    A `StochasticTensor` should be created by methods of a `Distribution`,
    and represents a sampled or observed random variable in the model.
    Although it stands for a random variable, it is actually a standard
    tensor in TensorFlow graph, where the randomness is achieved by random
    sampling.  That is to say, every `StochasticTensor` is equivalent
    to a set of random samples from the given `distribution`.
    The samples are stored so that every stochastic tensor should
    always represent the same set of samples.

    Parameters
    ----------
    distribution : Distribution | () -> Distribution
        The distribution that derives this stochastic tensor.

    samples : tf.Tensor | np.ndarray | float | int
        The samples of this stochastic tensor.

    observed : tf.Tensor | np.ndarray | float | int
        The observation of this stochastic tensor.
        One and only one of `samples`, `observed` should be specified.

    group_event_ndims : int
        If specify, override the default `group_event_ndims` of `distribution`.
        This argument can further be overrided by `group_event_ndims` argument
        of `prob` and `log_prob` method.
    """

    def __init__(self, distribution, samples=None, observed=None,
                 group_event_ndims=None):
        if (samples is not None and observed is not None) or \
                (samples is None and observed is None):
            raise ValueError('One and only one of `samples`, `observed` '
                             'should be specified.')
        elif samples is not None:
            tensor = samples
            is_observed = False
        else:
            tensor = observed
            is_observed = True

        if isinstance(tensor, StochasticTensor):
            tensor = tensor.__wrapped__
        if not isinstance(tensor, tf.Tensor):
            tensor = tf.convert_to_tensor(tensor)

        if not isinstance(distribution, Distribution):
            raise TypeError('`distribution` is expected to be a Distribution '
                            'but got %r.' % (distribution,))

        self.__wrapped__ = tensor
        self._self_is_observed = is_observed
        self._self_distrib = distribution
        self._self_group_event_ndims = group_event_ndims

    def __repr__(self):
        return 'StochasticTensor(%r)' % (self.__wrapped__,)

    def __hash__(self):
        # Necessary to support Python's collection membership operators
        return id(self)

    def __eq__(self, other):
        # Necessary to support Python's collection membership operators
        return id(self) == id(other)

    def __getattr__(self, name):
        return getattr(self.__wrapped__, name)

    def __setattr__(self, name, value):
        if name.startswith('_self_') or name == '__wrapped__':
            object.__setattr__(self, name, value)
        elif hasattr(type(self), name):
            object.__setattr__(self, name, value)
        else:
            setattr(self.__wrapped__, name, value)

    def __delattr__(self, name):
        if name.startswith('_self_'):
            object.__delattr__(self, name)
        elif name == '__wrapped__':
            raise TypeError('__wrapped__ must be an object')
        elif hasattr(type(self), name):
            object.__delattr__(self, name)
        else:
            delattr(self.__wrapped__, name)

    def _as_graph_element(self, allow_tensor=True, allow_operation=True):
        # this method brings support to ``session.run(...)`` and other
        # related methods which expects a graph element.
        if not allow_tensor:
            raise RuntimeError('`allow_tensor` is False but `StochasticTensor` '
                               'can only be casted to a `tf.Tensor`.')
        return self.__wrapped__

    @property
    def is_observed(self):
        """Whether or not this stochastic tensor is observed?"""
        return self._self_is_observed

    @property
    def group_event_ndims(self):
        """Get the number of dimensions to be considered as events group.

        Returns
        -------
        int | None
            The number of dimensions.  If `group_event_ndims` is not
            specified in the constructor, will return None.
        """
        return self._self_group_event_ndims

    @property
    def distribution(self):
        """Get the distribution that derives this stochastic tensor."""
        return self._self_distrib

    @property
    def is_continuous(self):
        """Whether or not the distribution is continuous?"""
        return self.distribution.is_continuous

    @property
    def is_reparameterized(self):
        """Whether or not the distribution is re-parameterized?"""
        return self.distribution.is_reparameterized

    @property
    def is_enumerable(self):
        """Whether or not the distribution is enumerable?

        A distribution with a finite value range is enumerable.
        Enumerable distribution could derive a special set of "samples",
        such that the probability of every possible value against each
        individual set of parameters could be computed.

        See Also
        --------
        Distribution.enum_sample
        """
        return self.distribution.is_enumerable

    @property
    def enum_value_count(self):
        """Get the count of possible values from the distribution.

        Returns
        -------
        int | tf.Tensor | None
            Static or dynamic count of possible values.
            If the distribution is not enumerable, it should return None.
        """
        return self.distribution.enum_value_count

    def log_lower_bound(self, group_event_ndims=None, name=None):
        return self.log_prob(
            group_event_ndims=group_event_ndims,
            name=name or 'log_lower_bound'
        )

    def log_prob(self, group_event_ndims=None, name=None):
        """Compute the log-probability of this stochastic tensor.

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
            The log-probability of this stochastic tensor.
        """
        if group_event_ndims is None:
            group_event_ndims = self.group_event_ndims
        return self.distribution.log_prob(
            self.__wrapped__, group_event_ndims=group_event_ndims,
            name=name
        )

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
            group_event_ndims = self.group_event_ndims
        return self.distribution.prob(
            self.__wrapped__, group_event_ndims=group_event_ndims,
            name=name
        )


def _to_tensor(value, dtype=None, name=None, as_ref=False):
    if dtype and not dtype.is_compatible_with(value.dtype):
        raise ValueError('Incompatible type conversion requested to type '
                         '%s for tensor of type %s' %
                         (dtype.name, value.dtype.name))
    if as_ref:
        raise ValueError('%r: Ref type not supported.' % value)
    return value.__wrapped__

tf.register_tensor_conversion_function(StochasticTensor, _to_tensor)
