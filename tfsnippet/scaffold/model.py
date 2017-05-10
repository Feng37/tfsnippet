# -*- coding: utf-8 -*-
import os
import shutil

import numpy as np
import six
import tensorflow as tf

from tfsnippet.utils import (ensure_variables_initialized,
                             VarScopeObject,
                             get_variables_as_dict,
                             reopen_variable_scope,
                             VariableSaver)

__all__ = ['Model']


class Model(VarScopeObject):
    """Scaffold for defining models with TensorFlow.

    This class provides a basic scaffold for defining models with TensorFlow.
    It settles a common practice for organizing the variables of parameters,
    as well as variables for training and prediction

    Parameters
    ----------
    name : str
        Name of the model.

        Note that if `name` is specified, the variable scope of constructed
        model will take exactly this name, even if another model has already
        taken such name.  That is to say, these two models will share the same
        variable scope.

    default_name : str
        Default name of the model.

        When `name` is not specified, the model will obtain a variable scope
        with unique name, according to the name suggested by `default_name`.
        If `default_name` is also not specified, it will use the underscored
        class name as the default name.

    global_step : tf.Variable
        Manually specify a `global_step` variable instead of using the one
        which is created inside the model variable scope.
    """

    def __init__(self, name=None, default_name=None, global_step=None):
        super(Model, self).__init__(name=name, default_name=default_name)

        # whether or not the model has been built?
        self._has_built = False

        # the global step variable of this model.
        self._global_step = global_step     # type: tf.Variable

    @property
    def has_built(self):
        """Whether or not the model has been built?"""
        return self._has_built

    def set_global_step(self, global_step):
        """Manually set a variable as global step.

        This method must be called before `build` or any other method
        that will cause the model to be built has been called.

        Parameters
        ----------
        global_step : tf.Variable
            The global step variable to use.

        Raises
        ------
        RuntimeError
            If the `global_step` has already been set or created.
        """
        if self._global_step is not None:
            raise RuntimeError(
                'The `global_step` variable has already been set or created.')
        self._global_step = global_step

    def get_global_step(self):
        """Get the global step variable of this model."""
        self.build()
        return self._global_step

    def build(self):
        """Build the model and the trainer.

        Although this method will be called automatically when the model
        is required to be built, however, it is recommended to call this
        method soon after the model object is constructed.
        """
        if self._has_built:
            return
        self._has_built = True

        with reopen_variable_scope(self.variable_scope):
            # create the global step variable if there's none
            if self._global_step is None:
                self._global_step = tf.get_variable(
                    'global_step', dtype=tf.int64, trainable=False,
                    initializer=np.asarray(0, dtype=np.int64)
                )

            # build the model
            self._build()

    def _build(self):
        """Build the main part of the model.

        Derived classes should override this to actually build the model.
        Note that all the variables to be considered as parameters must
        be created within the "model" sub-scope, which will be saved and
        restored via the interfaces of this class.

        For example, one may override the `_build` method as follows:

            class MyModel(Model):

                def _build(self):
                    with tf.variable_scope('model'):
                        hidden_layer = layers.fully_connected(
                            self.input_x, num_outputs=100
                        )
                        self.output = layers.fully_connected(
                            hidden_layer, num_outputs=1
                        )
                    with tf.variable_scope('train'):
                        loss = tf.reduce_mean(
                            tf.square(self.output - self.label)
                        )
                        self.train_op = tf.AdamOptimizer().minimize(loss)

        In this way, any variables introduced by the trainer will not be
        collected as parameter variables (although they will still be
        considered as model variables, and can be collected by the method
        `get_variables`).
        """
        raise NotImplementedError()

    def get_variables(self, collection=tf.GraphKeys.GLOBAL_VARIABLES):
        """Get the variables defined within the model's scope.

        This method will get variables which are in `variable_scope`
        and the specified `collection`, as a dict which maps relative
        names to variable objects.  By "relative name" we mean to remove
        the name of `variable_scope` from the front of variable names.

        Parameters
        ----------
        collection : str
            The name of the variable collection.
            If not specified, will use `tf.GraphKeys.GLOBAL_VARIABLES`.

        Returns
        -------
        dict[str, tf.Variable]
            Dict which maps from relative names to variable objects.
        """
        self.build()
        return get_variables_as_dict(self.variable_scope, collection=collection)

    def get_param_variables(self, collection=tf.GraphKeys.GLOBAL_VARIABLES):
        """Get the parameter variables.

        The parameter variables are the variables defined in "model"
        sub-scope within the model's variable scope.

        Parameters
        ----------
        collection : str
            The name of the variable collection.
            If not specified, will use `tf.GraphKeys.GLOBAL_VARIABLES`.

        Returns
        -------
        dict[str, tf.Variable]
            Dict which maps from relative names to variable objects.
        """
        self.build()
        vs_name = self.variable_scope.name + '/'
        if vs_name and not vs_name.endswith('/'):
            vs_name += '/'
        vs_name += 'model/'
        variables = get_variables_as_dict(vs_name, collection=collection)
        return {'model/' + k: v for k, v in six.iteritems(variables)}

    def ensure_variables_initialized(self):
        """Initialize all uninitialized variables within model's scope.

        If the `global_step` is created by the model itself, then it will
        also be initialized.  Otherwise the `global_step` must be manually
        initialized.
        """
        self.build()
        var_list = list(six.itervalues(self.get_variables()))
        ensure_variables_initialized(var_list)

    def save_model(self, save_dir, overwrite=False):
        """Save the model parameters onto disk.

        Parameters
        ----------
        save_dir : str
            Directory where to place the saved variables.

        overwrite : bool
            Whether or not to overwrite the existing directory?
        """
        self.build()
        path = os.path.abspath(save_dir)
        if os.path.exists(path):
            if overwrite:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            elif not os.path.isdir(path) or len(os.listdir(path)) > 0:
                raise IOError('%r already exists.' % save_dir)
        saver = VariableSaver(self.get_param_variables(), path)
        saver.save()

    def load_model(self, save_dir):
        """Load the model parameters from disk.

        Parameters
        ----------
        save_dir : str
            Directory where the saved variables are placed.
        """
        self.build()
        path = os.path.abspath(save_dir)
        saver = VariableSaver(self.get_param_variables(), path)
        saver.restore(ignore_non_exist=False)
