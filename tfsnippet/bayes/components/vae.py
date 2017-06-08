# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.utils import (VarScopeObject,
                             instance_reuse,
                             maybe_explicit_broadcast,
                             NOT_SPECIFIED)
from ..distributions import Distribution
from ..layers import StochasticLayer
from ..stochastic import StochasticObject, StochasticTensor
from ..utils import gather_log_lower_bound
from ..variational import sgvb

__all__ = ['VAE', 'DerivedVAE']


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

    latent_axis : int | tf.Tensor | None
        The axis(es) of z samples in log lower-bound.
        None if there's no explicit sampling axis. (default None)
    """

    def __init__(self, vae, x=None, z=None, z_posterior=None, latent_axis=None):
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
        self._latent_axis = latent_axis

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
    def latent_axis(self):
        """The axis(es) of z samples in log lower-bound."""
        return self._latent_axis

    def log_lower_bound(self, reduce_latent_axis=True, name=None):
        """Get the log lower-bound of this derived variational auto-encoder.

        The sampling axis of log lower-bound will be averaged out, so that
        the log lower-bound would be pseudo :math:`log p(x)`.  To get the
        log lower-bound with un-reduced latent axis(es), you may call
        `VAE.compute_log_lower_bound` manually.

        Returns
        -------
        tf.Tensor
            The computed log lower-bound.
        """
        if self._log_lower_bound is None:
            with tf.name_scope(name, default_name='log_lower_bound'):
                if reduce_latent_axis:
                    latent_axis = self.latent_axis
                else:
                    latent_axis = None

                if self.z_posterior is not None:
                    if self.x is not None:
                        assert(self.z is not None)
                        ret = self.vae.variational_solver(
                            model=[self.x, self.z],
                            variational=[self.z_posterior],
                            latent_axis=latent_axis
                        )
                    else:
                        assert(self.z is None)
                        ret = self.z_posterior.log_prob()
                else:
                    ret = sum(gather_log_lower_bound([self.x, self.z]))
                    if latent_axis is not None:
                        ret = tf.reduce_mean(ret, axis=latent_axis)
            self._log_lower_bound = ret
        return self._log_lower_bound


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

    validate_shape : bool
        Whether or not to validate the shape of samples or observations?
        (default False)

    variational_solver
        The algorithm for deriving log lower-bound (default `sgvb`)

        A variational approximator is a function which accepts arguments
        (model, variational, latent_axis) and produces the log lower-bound
        given these arguments.

    z_samples : int | tf.Tensor | None
        The number of z samples to take.  (default None)

    name, default_name : str
        Optional name and default name of this VAE.
    """

    def __init__(self, x_net, x_layer, z_net, z_layer, z_prior,
                 validate_shape=False, variational_solver=sgvb, z_samples=None,
                 name=None, default_name=None):
        super(VAE, self).__init__(name=name, default_name=default_name)
        self._x_net = x_net
        self._x_layer = x_layer
        self._z_net = z_net
        self._z_layer = z_layer
        self._z_prior = z_prior
        self._validate_shape = validate_shape
        self._variational_solver = variational_solver
        self._z_samples = z_samples

    @property
    def variational_solver(self):
        """Get the algorithm for deriving log lower-bound."""
        return self._variational_solver

    @instance_reuse
    def model(self, z, y=None, x=None, z_samples=NOT_SPECIFIED,
              x_samples=None):
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
            Specify the number of samples to take for output x.
            (default None)

        Returns
        -------
        (StochasticTensor, StochasticTensor)
            The derived (z, x) stochastic tensors.
        """
        if z_samples is NOT_SPECIFIED:
            z_samples = self._z_samples
        z = self._z_prior.sample_or_observe(
            z_samples, observed=z, validate_shape=self._validate_shape,
            group_event_ndims=1, name='z'
        )

        if y is None:
            features = z
        else:
            features = tf.concat(
                maybe_explicit_broadcast(z, y, tail_no_broadcast_ndims=1),
                axis=-1
            )
        x_params = self._x_net(features)

        x = self._x_layer(
            x_params, n_samples=x_samples, observed=x, group_event_ndims=1,
            validate_shape=self._validate_shape
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
                maybe_explicit_broadcast(x, y, tail_no_broadcast_ndims=1),
                axis=-1
            )
        z_params = self._z_net(features)

        if z_samples is NOT_SPECIFIED:
            z_samples = self._z_samples
        return self._z_layer(
            z_params, n_samples=z_samples, observed=z, group_event_ndims=1,
            validate_shape=self._validate_shape
        )

    def _detect_latent_axis(self, z, z_samples):
        if z_samples is None:
            latent_axis = None
        else:
            z_ndims = z.get_shape().ndims
            if z_ndims == 0:
                raise ValueError('`z_samples` is specified, but the '
                                 'sampled z is 0-dimensional.')
            elif z_ndims is None:
                z_ndims_assertion = tf.assert_rank_at_least(
                    z, 1, message='`z_samples` is specified, but the '
                                  'sampled z is 0-dimensional'
                )
                with tf.control_dependencies([z_ndims_assertion]):
                    z_ndims = tf.rank(z)
            latent_axis = -z_ndims
        return latent_axis

    @instance_reuse
    def reconstruct(self, x, y=None, z_samples=NOT_SPECIFIED, x_samples=None,
                    observe_x=False, latent_axis=NOT_SPECIFIED):
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
            Specify the number of samples to take for x.
            (default None)

        observe_x : bool
            Whether or not to fix observed x in output `StochasticTensor`?

            If True, the output x `StochasticTensor` will observe `x`.
            Otherwise the output x will have random samples.
            (default False)

        latent_axis : int | tuple[int] | tf.Tensor | None
            The axis(es) to be considered as the sampling dimensions of z.

            If not specified, will be automatically inferred.
            Explicitly set to None will force to consider no dimension as
            latent axis(es).

        Returns
        -------
        DerivedVAE
            The derived `StochasticTensor` objects as well as the log
            lower-bound of this variational auto-encoder.
        """
        if z_samples is NOT_SPECIFIED:
            z_samples = self._z_samples

        z_posterior = self.variational(x, y=y, z_samples=z_samples)
        z, x = self.model(z_posterior, y=y, x=x if observe_x else None,
                          z_samples=z_samples, x_samples=x_samples)

        if latent_axis is NOT_SPECIFIED:
            latent_axis = self._detect_latent_axis(z, z_samples)

        return DerivedVAE(self, x=x, z=z, z_posterior=z_posterior,
                          latent_axis=latent_axis)
