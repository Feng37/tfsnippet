# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tfsnippet.utils import (get_preferred_tensor_dtype,
                             reopen_variable_scope,
                             is_deterministic_shape,
                             maybe_explicit_broadcast)
from .base import Distribution

__all__ = [
    'Categorical',
    'OneHotCategorical',
    'Discrete',
    'OneHotDiscrete',
]


class _BaseCategorical(Distribution):
    """Base categorical distribution.

    Parameters
    ----------
    logits : tf.Tensor | np.ndarray
        A float tensor of shape (..., n_categories), which is the
        un-normalized log-odds of probabilities of the categories.

    probs : tf.Tensor | np.ndarray
        A float tensor of shape (..., n_categories), which is the
        normalized probabilities of the categories.

        One and only one of `logits` and `probs` should be specified.
        The relationship between these two arguments, if given each other,
        is stated as follows:

            .. math::
                \\begin{aligned}
                    \\text{logits} &= \\log (\\text{probs}) \\\\
                     \\text{probs} &= \\text{softmax} (\\text{logits})
                \\end{aligned}

    group_event_ndims : int | tf.Tensor
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

    check_numerics : bool
        Whether or not to check numerical issues? (default False)

    name : str
        Name of this normal distribution.

    default_name : str
        Default name of this normal distribution.
    """

    def __init__(self, logits=None, probs=None, group_event_ndims=None,
                 check_numerics=False, name=None, default_name=None):
        # check the arguments
        if (logits is None and probs is None) or \
                (logits is not None and probs is not None):
            raise ValueError('One and only one of `logits`, `probs` should '
                             'be specified.')

        if logits is not None:
            param_dtype = get_preferred_tensor_dtype(logits)
        else:
            param_dtype = get_preferred_tensor_dtype(probs)
        if not param_dtype.is_floating:
            raise TypeError('Categorical distribution parameters must be float '
                            'numbers.')

        super(_BaseCategorical, self).__init__(
            group_event_ndims=group_event_ndims,
            check_numerics=check_numerics,
            name=name,
            default_name=default_name
        )

        with reopen_variable_scope(self.variable_scope):
            with tf.name_scope('init'):
                # obtain parameter tensors
                if logits is not None:
                    logits = tf.convert_to_tensor(logits, dtype=param_dtype)
                    probs = tf.nn.softmax(logits, name='probs_given_logits')
                    probs_clipped = probs
                    probs_is_derived = True
                else:
                    probs = tf.convert_to_tensor(probs, dtype=param_dtype)
                    probs_eps = (
                        1e-11 if probs.dtype == tf.float64 else 1e-7)
                    probs_clipped = tf.clip_by_value(
                        probs, probs_eps, 1 - probs_eps
                    )
                    logits = self._check_numerics(
                        tf.log(probs, name='logits_given_probs'),
                        'logits'
                    )
                    probs_is_derived = False
                self._logits = logits
                self._probs = probs
                self._probs_is_derived = probs_is_derived
                self._probs_clipped = probs_clipped

                # derive the shape and data types of parameters
                logits_shape = logits.get_shape()
                self._static_batch_shape = logits_shape[:-1]
                if is_deterministic_shape(self._static_batch_shape):
                    self._dynamic_batch_shape = tf.constant(
                        self._static_batch_shape.as_list(),
                        dtype=tf.int32
                    )
                else:
                    self._dynamic_batch_shape = tf.shape(logits)[:-1]

                # infer the number of categories
                self._n_categories = logits_shape[-1].value
                if self._n_categories is None:
                    self._n_categories = tf.shape(logits)[-1]

    @property
    def param_dtype(self):
        return self._logits.dtype

    @property
    def is_continuous(self):
        return False

    @property
    def is_reparameterized(self):
        return False

    @property
    def is_enumerable(self):
        return True

    @property
    def enum_value_count(self):
        return self._n_categories

    @property
    def dynamic_batch_shape(self):
        return self._dynamic_batch_shape

    @property
    def static_batch_shape(self):
        return self._static_batch_shape

    @property
    def n_categories(self):
        """Get the number of categories.

        Returns
        -------
        int | tf.Tensor
            Constant or dynamic number of categories.
        """
        return self._n_categories

    @property
    def logits(self):
        """Get the un-normalized log-odds of Categorical distribution."""
        return self._logits

    @property
    def probs(self):
        """Get the probabilities of Categorical distribution."""
        return self._probs

    def _sample_n_sparse(self, n, dtype=None):
        # flatten the logits for feeding into `tf.multinomial`.
        if self.logits.get_shape().ndims == 2:
            logits_2d = self.logits
        else:
            logits_2d = tf.reshape(
                self.logits,
                tf.stack([-1, tf.shape(self.logits)[-1]])
            )
            logits_2d.set_shape(
                tf.TensorShape([None]).concatenate(
                    self.logits.get_shape()[-1:]))

        # derive the samples
        samples = tf.multinomial(logits_2d, n)
        if dtype is not None:
            samples = tf.cast(samples, dtype=dtype)

        # reshape back into the requested shape
        samples = tf.transpose(samples)
        if is_deterministic_shape([n]):
            static_shape = tf.TensorShape([n])
        else:
            static_shape = tf.TensorShape([None])
        static_shape = static_shape.concatenate(self.logits.get_shape()[:-1])
        dynamic_shape = tf.concat([[n], tf.shape(self.logits)[:-1]], axis=0)
        samples = tf.reshape(samples, dynamic_shape)
        samples.set_shape(static_shape)
        return samples

    def _enum_values_sparse(self, dtype=None):
        # reshape the enumerated values to match the batch shape.
        reshape_shape = tf.concat(
            [[self.n_categories],
             tf.ones(tf.stack([tf.size(self.dynamic_batch_shape)]),
                     dtype=tf.int32)],
            axis=0
        )
        samples = tf.reshape(tf.range(0, self.n_categories, dtype=dtype),
                             reshape_shape)

        # tile the enumerated values along batch shape
        tile_shape = tf.concat(
            [[1], self.dynamic_batch_shape], axis=0)
        samples = tf.tile(samples, tile_shape)

        # fix the static shape of the samples
        if is_deterministic_shape([self.n_categories]):
            static_shape = tf.TensorShape([self.n_categories])
        else:
            static_shape = tf.TensorShape([None])
        samples.set_shape(static_shape.concatenate(self.static_batch_shape))
        return samples

    def _analytic_kld(self, other):
        if isinstance(other, _BaseCategorical):
            # sum(p * log(p / q))
            probs1 = self.probs
            if self._probs_is_derived:
                log_probs1 = tf.nn.log_softmax(self.logits)
            else:
                log_probs1 = self._check_numerics(
                    tf.log(self._probs_clipped), 'log(p)'
                )

            if other._probs_is_derived:
                log_probs2 = tf.nn.log_softmax(other.logits)
            else:
                log_probs2 = other._check_numerics(
                    tf.log(other._probs_clipped), 'log(p)')

            return tf.reduce_sum(
                probs1 * (log_probs1 - log_probs2), axis=-1)


class Categorical(_BaseCategorical):
    """Categorical distribution.

    Parameters
    ----------
    logits : tf.Tensor | np.ndarray
        A float tensor of shape (..., n_categories), which is the
        un-normalized log-odds of probabilities of the categories.

    probs : tf.Tensor | np.ndarray
        A float tensor of shape (..., n_categories), which is the
        normalized probabilities of the categories.

        One and only one of `logits` and `probs` should be specified.
        The relationship between these two arguments, if given each other,
        is stated as follows:

            .. math::
                \\begin{aligned}
                    \\text{logits} &= \\log (\\text{probs}) \\\\
                     \\text{probs} &= \\text{softmax} (\\text{logits})
                \\end{aligned}

    dtype : tf.DType | np.dtype | str
        The data type of samples from the distribution. (default is `tf.int32`)

    group_event_ndims : int | tf.Tensor
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

    check_numerics : bool
        Whether or not to check numerical issues? (default False)

    name : str
        Name of this normal distribution.

    default_name : str
        Default name of this normal distribution.
    """

    def __init__(self, logits=None, probs=None, dtype=None,
                 group_event_ndims=None, check_numerics=False,
                 name=None, default_name=None):
        if dtype is None:
            dtype = tf.int32
        else:
            dtype = tf.as_dtype(dtype)

        super(Categorical, self).__init__(
            logits=logits,
            probs=probs,
            group_event_ndims=group_event_ndims,
            check_numerics=check_numerics,
            name=name,
            default_name=default_name
        )
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def dynamic_value_shape(self):
        return tf.constant([], dtype=tf.int32)

    @property
    def static_value_shape(self):
        return tf.TensorShape([])

    def _sample_n(self, n):
        return self._sample_n_sparse(n, self.dtype)

    def _enum_values(self):
        return self._enum_values_sparse(self.dtype)

    def _log_prob_with_logits(self, x):
        x = tf.convert_to_tensor(x)
        if not x.dtype.is_integer:
            x = tf.cast(x, dtype=tf.int32)

        if self.logits.get_shape()[:-1] == x.get_shape():
            logits = self.logits
        else:
            logits = self.logits * tf.ones_like(
                tf.expand_dims(x, -1), dtype=self.logits.dtype)
            logits_shape = tf.shape(logits)[:-1]
            x *= tf.ones(logits_shape, dtype=x.dtype)
            x.set_shape(tf.TensorShape(logits.get_shape()[:-1]))

        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=x, logits=logits,
        )

    def _log_prob_with_probs(self, x):
        x = tf.one_hot(
            tf.cast(x, dtype=tf.int32),
            self.n_categories,
            dtype=self.param_dtype
        )
        log_p = self._check_numerics(
            tf.log(self._probs_clipped),
            'log(p)'
        )
        return tf.reduce_sum(x * log_p, axis=-1)

    def _log_prob(self, x):
        if self._probs_is_derived:
            return self._log_prob_with_logits(x)
        else:
            # TODO: check whether this is better than using derived logits.
            return self._log_prob_with_probs(x)

Discrete = Categorical


class OneHotCategorical(_BaseCategorical):
    """Categorical distribution with one-hot encoded samples.

    Parameters
    ----------
    logits : tf.Tensor | np.ndarray
        A float tensor of shape (..., n_categories), which is the
        un-normalized log-odds of probabilities of the categories.

    probs : tf.Tensor | np.ndarray
        A float tensor of shape (..., n_categories), which is the
        normalized probabilities of the categories.

        One and only one of `logits` and `probs` should be specified.
        The relationship between these two arguments, if given each other,
        is stated as follows:

            .. math::
                \\begin{aligned}
                    \\text{logits} &= \\log (\\text{probs}) \\\\
                     \\text{probs} &= \\text{softmax} (\\text{logits})
                \\end{aligned}

    dtype : tf.DType | np.dtype | str
        The data type of samples from the distribution. (default is `tf.int32`)

    group_event_ndims : int | tf.Tensor
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

    check_numerics : bool
        Whether or not to check numerical issues? (default False)

    name : str
        Name of this normal distribution.

    default_name : str
        Default name of this normal distribution.
    """

    def __init__(self, logits=None, probs=None, dtype=None,
                 group_event_ndims=None, check_numerics=False,
                 name=None, default_name=None):
        if dtype is None:
            dtype = tf.int32
        else:
            dtype = tf.as_dtype(dtype)

        super(OneHotCategorical, self).__init__(
            logits=logits,
            probs=probs,
            group_event_ndims=group_event_ndims,
            check_numerics=check_numerics,
            name=name,
            default_name=default_name
        )
        self._dtype = dtype

        with reopen_variable_scope(self.variable_scope):
            with tf.name_scope('init'):
                # derive the value shape of parameters
                logits_shape = self.logits.get_shape()
                self._static_value_shape = logits_shape[-1:]
                if is_deterministic_shape(self._static_value_shape):
                    self._dynamic_value_shape = tf.constant(
                        self._static_value_shape.as_list(),
                        dtype=tf.int32
                    )
                else:
                    self._dynamic_value_shape = tf.shape(logits)[-1:]

    @property
    def dtype(self):
        return self._dtype

    @property
    def dynamic_value_shape(self):
        return self._dynamic_value_shape

    @property
    def static_value_shape(self):
        return self._static_value_shape

    def _sample_n(self, n):
        samples = self._sample_n_sparse(n, self.dtype)
        samples = tf.one_hot(samples, self.n_categories, dtype=self.dtype)
        return samples

    def _enum_values(self):
        samples = self._enum_values_sparse(self.dtype)
        samples = tf.one_hot(samples, self.n_categories, dtype=self.dtype)
        return samples

    def _log_prob_with_logits(self, x):
        # flatten `x` and `logits` as requirement of
        # `tf.nn.softmax_cross_entropy_with_logits`
        x = tf.cast(x, dtype=self.param_dtype)
        x, logits = maybe_explicit_broadcast(x, self.logits)

        if x.get_shape().ndims == 2:
            x_2d, logits_2d = x, logits
        else:
            dynamic_shape = tf.stack([-1, tf.shape(x)[-1]], axis=0)
            x_2d = tf.reshape(x, dynamic_shape)
            logits_2d = tf.reshape(logits, dynamic_shape)

            static_shape = (
                tf.TensorShape([None]).concatenate(x.get_shape()[-1:]))
            x_2d.set_shape(static_shape)
            logits_2d.set_shape(static_shape)

        # derive the flatten log p(x)
        log_p_2d = -tf.nn.softmax_cross_entropy_with_logits(
            labels=x_2d, logits=logits_2d,
        )

        # reshape log p(x) back into desired shape
        if x.get_shape().ndims == 2:
            log_p = log_p_2d
        else:
            static_shape = x.get_shape()[: -1]
            log_p = tf.reshape(log_p_2d, tf.shape(x)[: -1])
            log_p.set_shape(static_shape)

        return log_p

    def _log_prob_with_probs(self, x):
        x = tf.cast(x, dtype=self.param_dtype)
        log_p = self._check_numerics(
            tf.log(self._probs_clipped),
            'log(p)'
        )
        return tf.reduce_sum(x * log_p, axis=-1)

    def _log_prob(self, x):
        if self._probs_is_derived:
            return self._log_prob_with_logits(x)
        else:
            # TODO: check whether this is better than using derived logits.
            return self._log_prob_with_probs(x)

OneHotDiscrete = OneHotCategorical
