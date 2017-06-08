# -*- coding: utf-8 -*-
import six
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

    def log_lower_bound(self, reduce_latent_axis=True, name=None):
        """Compute the log-probability lower-bound.

        Parameters
        ----------
        name : str
            Optional name of this operation.

        reduce_latent_axis : bool
            Whether or not to average out the log lower-bounds along
            the sampling dimensions of latent variables?
            (default True)

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
    Although it stands for a random variable, it is actually acts as a
    wrapper for an instance of `tf.Tensor`, where the randomness is achieved
    by random sampling.  All the attributes and methods of the wrapped
    `tf.Tensor` could be accessed through `StochasticTensor` object.

    Parameters
    ----------
    distribution : Distribution | () -> Distribution
        The distribution that derives this stochastic tensor.

    samples : tf.Tensor | np.ndarray
        The samples of this stochastic tensor.

    observed : tf.Tensor | np.ndarray
        The observation of this stochastic tensor.  Data type of this
        tensor will be casted to `distribution.dtype`.

        One and only one of `samples`, `observed` should be specified.

    samples_ndims : tf.Tensor | int
        The number of sampling dimensions of this stochastic tensor.

        If not specified, then no dimension would be considered as the
        sampling dimension (i.e., only one sample without explicit
        dimension for samples).  (default None)

    group_event_ndims : int | tf.Tensor
        If specified, override the default `group_event_ndims` of
        `distribution`.  (default None)

    validate_shape : bool
        Whether or not to validate the shape of samples or observations?
        See `Distribution.validate_samples_shape` for more details.
        (default False)
    """

    def __init__(self, distribution, samples=None, observed=None,
                 samples_ndims=None, group_event_ndims=None,
                 validate_shape=False):
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

        if not isinstance(distribution, Distribution):
            raise TypeError('`distribution` is expected to be a Distribution '
                            'but got %r.' % (distribution,))

        if isinstance(tensor, StochasticTensor):
            tensor = tensor.__wrapped__
        if not isinstance(tensor, tf.Tensor):
            tensor = tf.convert_to_tensor(tensor, distribution.dtype)
        if tensor.dtype != distribution.dtype:
            tensor = tf.cast(tensor, dtype=distribution.dtype)

        if validate_shape:
            tensor = distribution.validate_samples_shape(tensor)

        self.__wrapped__ = tensor
        self._self_is_observed = is_observed
        self._self_distrib = distribution
        self._self_samples_ndims = samples_ndims
        self._self_group_event_ndims = group_event_ndims
        self._self_prob = None      # cached prob
        self._self_log_prob = None  # cached log-prob

    def __repr__(self):
        return 'StochasticTensor(%r)' % (self.__wrapped__,)

    def __hash__(self):
        # Necessary to support Python's collection membership operators
        return id(self)

    def __eq__(self, other):
        # Necessary to support Python's collection membership operators
        return id(self) == id(other)

    @property
    def is_observed(self):
        """Whether or not this stochastic tensor is observed?"""
        return self._self_is_observed

    @property
    def samples_ndims(self):
        """Get the number of sampling dimensions of this stochastic tensor.

        Returns
        -------
        int | tf.Tensor | None
            The number of dimensions.  If `samples_ndims` is not
            specified in the constructor, will return None.
        """
        return self._self_samples_ndims

    @property
    def group_event_ndims(self):
        """Get the number of dimensions to be considered as events group.

        Returns
        -------
        int | tf.Tensor | None
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

    def log_lower_bound(self, reduce_latent_axis=True, name=None):
        return self.log_prob(name=name or 'log_lower_bound')

    def log_prob(self, name=None):
        """Compute the log-probability of this stochastic tensor.

        Parameters
        ----------
        name : str
            Optional name of this operation.

        Returns
        -------
        tf.Tensor
            The log-probability of this stochastic tensor.
        """
        if self._self_log_prob is None:
            self._self_log_prob = self.distribution.log_prob(
                self.__wrapped__, group_event_ndims=self.group_event_ndims,
                name=name
            )
        return self._self_log_prob

    def prob(self, name=None):
        """Compute the likelihood of this stochastic tensor.

        Parameters
        ----------
        name : str
            Optional name of this operation.

        Returns
        -------
        tf.Tensor
            The likelihood of this stochastic tensor.
        """
        if self._self_prob is None:
            if self.distribution.has_specialized_prob_method:
                self._self_prob = self.distribution.prob(
                    self.__wrapped__,
                    group_event_ndims=self.group_event_ndims,
                    name=name
                )
            else:
                self._self_prob = tf.exp(self.log_prob(), name=name or 'prob')
                # mimic the true behavior of distributions without
                # specialized prob method
                self._self_prob = self.distribution._check_numerics(
                    self._self_prob, 'prob'
                )
        return self._self_prob

    # mimic `tf.Tensor` interface
    def __dir__(self):
        if six.PY3:
            ret = object.__dir__(self)
        else:
            # code is based on
            # http://www.quora.com/How-dir-is-implemented-Is-there-any-PEP-related-to-that
            def get_attrs(obj):
                import types
                if not hasattr(obj, '__dict__'):
                    return []  # slots only
                if not isinstance(obj.__dict__, (dict, types.DictProxyType)):
                    raise TypeError("%s.__dict__ is not a dictionary"
                                    "" % obj.__name__)
                return obj.__dict__.keys()

            def dir2(obj):
                attrs = set()
                if not hasattr(obj, '__bases__'):
                    # obj is an instance
                    if not hasattr(obj, '__class__'):
                        # slots
                        return sorted(get_attrs(obj))
                    klass = obj.__class__
                    attrs.update(get_attrs(klass))
                else:
                    # obj is a class
                    klass = obj

                for cls in klass.__bases__:
                    attrs.update(get_attrs(cls))
                    attrs.update(dir2(cls))
                attrs.update(get_attrs(obj))
                return list(attrs)

            ret = dir2(self)

        ret = list(set(dir(self.__wrapped__) + ret))
        return ret

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
        elif hasattr(type(self), name):
            object.__delattr__(self, name)
        else:
            delattr(self.__wrapped__, name)

    def __iter__(self):
        raise TypeError('`StochasticTensor` object is not iterable.')

    def __bool__(self):
        raise TypeError(
            'Using a `StochasticTensor` as a Python `bool` is not allowed. '
            'Use `if t is not None:` instead of `if t:` to test if a '
            'tensor is defined, and use TensorFlow ops such as '
            'tf.cond to execute subgraphs conditioned on the value of '
            'a tensor.'
        )

    def __nonzero__(self):
        raise TypeError(
            'Using a `StochasticTensor` as a Python `bool` is not allowed. '
            'Use `if t is not None:` instead of `if t:` to test if a '
            'tensor is defined, and use TensorFlow ops such as '
            'tf.cond to execute subgraphs conditioned on the value of '
            'a tensor.'
        )

    def _as_graph_element(self, allow_tensor=True, allow_operation=True):
        # this method brings support to ``session.run(...)`` and other
        # related methods which expects a graph element.
        if not allow_tensor:
            raise RuntimeError('Can not convert a Tensor into a Operation.')
        return self.__wrapped__


def _to_tensor(value, dtype=None, name=None, as_ref=False):
    if dtype and not dtype.is_compatible_with(value.dtype):
        raise ValueError('Incompatible type conversion requested to type '
                         '%s for tensor of type %s' %
                         (dtype.name, value.dtype.name))
    if as_ref:
        raise ValueError('%r: Ref type not supported.' % value)
    return value.__wrapped__

tf.register_tensor_conversion_function(StochasticTensor, _to_tensor)
