# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.utils import (VarScopeObject,
                             instance_reuse,
                             maybe_explicit_broadcast)
from ..distributions import Distribution
from ..layers import StochasticLayer
from ..stochastic import StochasticObject, StochasticTensor
from ..utils import gather_log_lower_bound
from ..variational import sgvb

__all__ = ['VAE', 'DerivedVAE']

NOT_SPECIFIED = object()


class DerivedVAE(StochasticObject):
    """Derived variational auto-encoder.

    This class holds the derived StochasticTensor objects (akka x and z),
    as well as the log lower-bound of a certain `VAE` object.

    Parameters
    ----------
    vae : VAE
        The variational auto-encoder that derives x, z and log lower-bound.

    x : StochasticTensor | None
        The output x `StochasticTensor`, or None not derived.

    z : StochasticTensor | None
        The latent z `StochasticTensor`, or None not derived.

    z_posterior : StochasticTensor | None
        The variational posterior z `StochasticTensor`, or None not derived.

    z_axis : int | tf.Tensor | None
        The axis of z samples.  None if no explicit sampling axis.
        (default None)
    """

    def __init__(self, vae, x=None, z=None, z_posterior=None, z_axis=None):
        if (z is None and x is not None) or (z is not None and x is None):
            raise ValueError('`z` and `x` must be both specified or neither '
                             'specified.')
        if x is None and z is None and z_posterior is None:
            raise ValueError('At least `x` and `z`, or `z_posterior` '
                             'should be specified.')
        self._vae = vae
        self._x = x
        self._z = z
        self._z_posterior = z_posterior
        self._z_axis = z_axis

        self._log_lower_bound = None  # cached log lower-bound

    @property
    def vae(self):
        """Get the `VAE` object."""
        return self._vae

    @property
    def x(self):
        """Get the output x `StochasticTensor` object."""
        return self._x

    @property
    def z(self):
        """Get the latent z `StochasticTensor` object."""
        return self._z

    @property
    def z_posterior(self):
        """Get the variational posterior z `StochasticTensor` object."""
        return self._z_posterior

    @property
    def z_axis(self):
        """Get the axis of z samples."""
        return self._z_axis

    def log_lower_bound(self, name=None):
        """Get the log lower-bound of this derived variational auto-encoder.

        The sampling axis of log lower-bound will be averaged out, so that
        the log lower-bound would be pseudo :math:`log p(x)`.  To get the
        log lower-bound with un-reduced sampling axis, you may call
        `VAE.compute_log_lower_bound` manually.

        Returns
        -------
        tf.Tensor
            The computed log lower-bound.
        """
        return self.vae.compute_log_lower_bound(self, reduce_z_axis=True)


class VAE(VarScopeObject):
    """General implementation of variational auto-encoder.

    This class provides a generation implementation of the variational
    auto-encoder, which takes two callable objects as the factory for
    generation and variational sub-network (given their inputs), as well
    as a distribution object as the prior for latent variable.

    Note that the `group_event_ndims` of x and z will always be set to 1.

    Parameters
    ----------
    x_net : (tf.Tensor) -> dict
        The generation network that produces distribution parameters for x.

    x_layer : StochasticLayer
        The `StochasticLayer` that derives x given distribution parameters.

        Note that `n_samples`, `observed` and `group_event_ndims` specified
        in the layer constructor will always be overwritten.

    z_net : (tf.Tensor) -> dict
        The variational inference network that produces distribution
        parameters for z.

    z_layer : StochasticLayer
        The `StochasticLayer` that derives z given distribution parameters.

        Note that `n_samples`, `observed` and `group_event_ndims` specified
        in the layer constructor will always be overwritten.

    z_prior : Distribution
        The prior distribution for z.

        Note that `group_event_ndims` will always be set to 1.

    lower_bound_algo
        The algorithm for deriving log lower-bound (default `sgvb`)

        A variational approximator is a function which accepts arguments
        (model, variational, latent_axis) and produces the log lower-bound
        given these arguments.

    z_samples : int | tf.Tensor | None
        The number of z samples to take.  (default None)

    x_samples : int | tf.Tensor | None
        The number of x samples to take.  (default None)

    name, default_name : str
        Optional name and default name of this VAE.
    """

    def __init__(self, x_net, x_layer, z_net, z_layer, z_prior,
                 lower_bound_algo=sgvb, z_samples=None, x_samples=None,
                 name=None, default_name=None):
        super(VAE, self).__init__(name=name, default_name=default_name)
        self._x_net = x_net
        self._x_layer = x_layer
        self._z_net = z_net
        self._z_layer = z_layer
        self._z_prior = z_prior
        self._lower_bound_algo = lower_bound_algo
        self._z_samples = z_samples
        self._x_samples = x_samples

    @instance_reuse
    def compute_log_lower_bound(self, derived, reduce_z_axis=True):
        """Compute the log lower-bound for derived VAE.

        If the derived VAE is a reconstructed VAE (i.e., z is sampled
        from :math:`q(z|x)`), then `lower_bound_algo` is used to derive
        the approximated log lower-bound.

        If `z_posterior` is not derived, while only x and z are given,
        then the joint log-probability of x and z are computed.

        Otherwise if only `z_posterior` is derived, then :math:`q(z|x)`
        is used as the log lower-bound.

        Parameters
        ----------
        derived : DerivedVAE
            The derived variational auto-encoder.

        reduce_z_axis : bool
            Whether or not to reduce the sampling dimension of z?
            (default True)

        Returns
        -------
        tf.Tensor
            The log lower-bound of the derived VAE.
        """
        if derived.z_posterior is not None:
            if derived.x is not None:
                assert(derived.z is not None)
                ret = self._lower_bound_algo(
                    model=[derived.x, derived.z],
                    variational=[derived.z_posterior],
                )
            else:
                assert(derived.z is None)
                ret = derived.z_posterior.log_prob()
        else:
            ret = sum(gather_log_lower_bound([derived.x, derived.z]))
        if derived.z_axis is not None and reduce_z_axis:
            ret = tf.reduce_mean(ret, axis=derived.z_axis)
        return ret

    @instance_reuse
    def model(self, z, y=None, x=None, z_samples=NOT_SPECIFIED,
              x_samples=NOT_SPECIFIED):
        """Derive the `StochasticTensor` objects of generation network.

        Parameters
        ----------
        z : tf.Tensor | StochasticTensor
            The latent variable samples.

        y : tf.Tensor
            Optional conditional input for CVAE.  See [1] for more details.

            [1] Lee, Honglak. "Tutorial on deep learning and applications."
                NIPS 2010 Workshop on Deep Learning and Unsupervised Feature
                Learning. 2010.

        x : tf.Tensor
            The observations for the output `StochasticTensor`. (default None)

        z_samples : int | tf.Tensor | None
            If specified, override `z_samples` specified in the constructor.

        x_samples : int | tf.Tensor | None
            If specified, override `x_samples` specified in the constructor.

        Returns
        -------
        (StochasticTensor, StochasticTensor)
            The derived (z, x) stochastic tensors.
        """
        if z_samples is NOT_SPECIFIED:
            z_samples = self._z_samples
        z = self._z_prior.sample_or_observe(
            z_samples, observed=z, group_event_ndims=1, name='z')

        if y is None:
            features = z
        else:
            features = tf.concat(
                maybe_explicit_broadcast(z, y, tail_no_broadcast_ndims=1),
                axis=-1
            )
        x_params = self._x_net(features)

        if x_samples is NOT_SPECIFIED:
            x_samples = self._x_samples
        x = self._x_layer(
            x_params, n_samples=x_samples, observed=x, group_event_ndims=1,
            name='x'
        )

        return z, x

    @instance_reuse
    def variational(self, x, y=None, z=None, z_samples=NOT_SPECIFIED):
        """Derive the `StochasticTensor` objects of variational network.

        Parameters
        ----------
        x : tf.Tensor
            The model inputs.

        y : tf.Tensor
            Optional conditional input for CVAE.  See [1] for more details.

            [1] Lee, Honglak. "Tutorial on deep learning and applications."
                NIPS 2010 Workshop on Deep Learning and Unsupervised Feature
                Learning. 2010.

        z : tf.Tensor | StochasticTensor
            The latent variable observations.  (default None)

        z_samples : int | tf.Tensor | None
            If specified, override `z_samples` specified in the constructor.

        Returns
        -------
        StochasticTensor
            The derived z stochastic tensors.
        """
        if y is None:
            features = x
        else:
            features = tf.concat(
                maybe_explicit_broadcast([x, y], tail_no_broadcast_ndims=1),
                axis=-1
            )
        z_params = self._z_net(features)

        if z_samples is NOT_SPECIFIED:
            z_samples = self._z_samples
        return self._z_layer(
            z_params, n_samples=z_samples, observed=z, group_event_ndims=1,
            name='z'
        )

    @instance_reuse(scope='reconstruct')
    def reconstruct(self, x, y=None, z_samples=NOT_SPECIFIED,
                    x_samples=NOT_SPECIFIED):
        """Derive the variational auto-encoder for x reconstruction.

        Parameters
        ----------
        x : tf.Tensor
            The model inputs, akka, x to be reconstructed.

        y : tf.Tensor
            Optional conditional input for CVAE.  See [1] for more details.

            [1] Lee, Honglak. "Tutorial on deep learning and applications."
                NIPS 2010 Workshop on Deep Learning and Unsupervised Feature
                Learning. 2010.

        z_samples : int | tf.Tensor | None
            If specified, override `z_samples` specified in the constructor.

        x_samples : int | tf.Tensor | None
            If specified, override `x_samples` specified in the constructor.

        Returns
        -------
        DerivedVAE
            The derived `StochasticTensor` objects as well as the log
            lower-bound of this variational auto-encoder.
        """
        z_posterior = self.variational(x, y=y, n_samples=z_samples)
        z, x = self.model(z_posterior, y=y, n_samples=x_samples)

        if z_samples is not None:
            # todo: get a better way to determine the z-axis
            z_ndims = z.get_shape().ndims
            if z_ndims is None:
                z_ndims = tf.size(tf.shape(z))
                with tf.control_dependencies([tf.assert_greater(z_ndims, 0)]):
                    z_ndims = tf.identity(z_ndims)
            else:
                assert(z_ndims != 0)
            z_ndims = -z_ndims
        else:
            z_ndims = None

        return DerivedVAE(self, x=x, z=z, z_axis=z_ndims)
