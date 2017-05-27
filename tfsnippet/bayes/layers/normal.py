# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, xavier_initializer

from ..distributions import Normal
from .base import StochasticLayer

__all__ = ['NormalLayer']


class NormalLayer(StochasticLayer):
    """Stochastic layer for univariate normal distribution.

    This layer will compute `mean` and `stddev`/`logstd` out of the layer
    input tensor by applying linear transformations.
    For `stddev`, `tf.nn.softplus` activation will be applied after the
    linear transformation, so as to ensure it be positive.

    Parameters
    ----------
    num_outputs : int | tf.Tensor
        Size of the last dimension in the layer output.

        All the former dimensions will keep the same shape as the
        input to this layer, except for the last dimension.

    use_stddev : bool
        Whether or not to use `stddev` instead of `logstd` for Normal
        distribution? (default False)

    stddev_min : float | tf.Tensor
        The minimum `stddev` of the Normal distribution. (default 1e-3)

        This number will be added to `stddev` computed by `tf.nn.softplus`
        activation function, after the linear transformation applied on
        the layer input.  It will take effect only if `use_stddev` is set
        to True.

    n_samples : int | tf.Tensor | None
        Generate this number of samples via `sample_n` method, unless
        `observed` is specified.  Default is None, indicating just one
        sample without the auxiliary sample dimension.

        See `Distribution.sample_n` and `Distribution.sample_or_observe`
        for more details.

    observed : tf.Tensor | np.ndarray | float | int
        The observations.  If specified, will use the observations as
        output, instead sampling from the distribution. (default None)

    group_event_ndims : int | tf.Tensor
        If specify, this number of dimensions at the end of `batch_shape`
        would be considered as a group of events, whose probabilities are
        to be accounted together. (default None)

    weights_initializer
        Initializers for the weights of the linear transformations.
        (default `tf.contrib.layers.xavier_initializer`)

    weights_regularizer
        Regularizers for the weights of the linear transformations.
        (default None)

    biases_initializer
        Initializers for the biases of the linear transformations.
        (default `tf.zero_initializer`)

    biases_regularizer
        Regularizers for the biases of the linear transformations.
        (default None)

    trainable : bool
        Whether or not this layer is trainable? (default True)

    name, default_name : str
        Name and default name of this stochastic layer.

    See Also
    --------
    StochasticLayer
    """

    def __init__(self,
                 num_outputs,
                 use_stddev=False,
                 stddev_min=1e-3,
                 n_samples=None,
                 observed=None,
                 group_event_ndims=None,
                 weights_initializer=xavier_initializer(),
                 biases_initializer=tf.zeros_initializer(),
                 weights_regularizer=None,
                 biases_regularizer=None,
                 trainable=True,
                 name=None,
                 default_name=None):
        super(NormalLayer, self).__init__(
            n_samples=n_samples,
            observed=observed,
            group_event_ndims=group_event_ndims,
            name=name,
            default_name=default_name,
        )
        self._num_outputs = num_outputs
        self._stddev_min = stddev_min
        self._use_stddev = use_stddev
        self._weights_initializer = weights_initializer
        self._biases_initializer = biases_initializer
        self._weights_regularizer = weights_regularizer
        self._biases_regularizer = biases_regularizer
        self._trainable = trainable

    @property
    def num_outputs(self):
        """Size of the last dimension in the layer output."""
        return self._num_outputs

    @property
    def use_stddev(self):
        """Whether or not to use `stddev` instead of `logstd`?"""
        return self._use_stddev

    @property
    def stddev_min(self):
        """The minimum `stddev` of the Normal distribution."""
        return self._stddev_min

    def _call(self, x, n_samples, observed, group_event_ndims):
        def dense(activation_fn=None, scope=None):
            return fully_connected(
                x,
                num_outputs=self.num_outputs,
                activation_fn=activation_fn,
                weights_initializer=self._weights_initializer,
                weights_regularizer=self._weights_regularizer,
                biases_initializer=self._biases_initializer,
                biases_regularizer=self._biases_regularizer,
                trainable=self._trainable,
                scope=scope,
            )

        # derive the parameters for Normal distribution
        mean = dense(scope='mean')
        stddev = logstd = None
        if self.use_stddev:
            stddev = self.stddev_min + dense(activation_fn=tf.nn.softplus,
                                             scope='stddev')
        else:
            logstd = dense(scope='logstd')

        # get the distribution instance
        distrib = Normal(mean, stddev=stddev, logstd=logstd,
                         group_event_ndims=group_event_ndims)

        # get the samples
        return distrib.sample_or_observe(n_samples, observed=observed)
