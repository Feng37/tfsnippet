# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.utils import (VarScopeObject,
                             instance_reuse,
                             maybe_explicit_broadcast)
from ..distributions import Distribution
from ..stochastic import StochasticObject, StochasticTensor
from ..layers import StochasticLayer
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

    log_lower_bound : tf.Tensor
        The log lower-bound of the derived variational auto-encoder.

        It may or may not contain the latent dimension, which depends
        on the arguments used to derive this variational auto-encoder.
    """

    def __init__(self, vae, x, z, z_posterior, log_lower_bound):
        self._vae = vae
        self._x = x
        self._z = z
        self._z_posterior = z_posterior
        self._log_lower_bound = log_lower_bound

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

    def log_lower_bound(self, name=None):
        """Get the log lower-bound of this derived variational auto-encoder."""
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

    approximator
        The variational approximator for deriving log lower-bound.
        (default `tfsnippet.bayes.sgvb`)

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
                 approximator=sgvb, z_samples=None, x_samples=None,
                 name=None, default_name=None):
        super(VAE, self).__init__(name=name, default_name=default_name)
        self._x_net = x_net
        self._x_layer = x_layer
        self._z_net = z_net
        self._z_layer = z_layer
        self._z_prior = z_prior
        self._approximator = approximator
        self._z_samples = z_samples
        self._x_samples = x_samples

    @instance_reuse(scope='model')
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

    @instance_reuse(scope='variational')
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
                    x_samples=NOT_SPECIFIED, reduce_latent=True):
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

        reduce_latent : bool
            Whether or not to reduce the latent dimension of log lower-bound?
            (default True)

        Returns
        -------
        DerivedVAE
            The derived `StochasticTensor` objects as well as the log
            lower-bound of this variational auto-encoder.
        """
        z_posterior = self.variational(x, y=y, n_samples=z_samples)
        z, x = self.model(z_posterior, y=y, n_samples=x_samples)

        if reduce_latent:
            if z_samples is None:
                latent_axis = None
            elif x_samples is not None:
                latent_axis = 1
            else:
                latent_axis = 0
        else:
            latent_axis = None
        log_lower_bound = self._approximator(
            model=[x, z],
            variational=[z_posterior],
            latent_axis=latent_axis,
        )

        return DerivedVAE(self, x=x, z=z, log_lower_bound=log_lower_bound)
