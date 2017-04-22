# -*- coding: utf-8 -*-
import numpy as np
import six
import tensorflow as tf

from tfsnippet.utils import (get_default_session_or_error,
                             ensure_variables_initialized,
                             open_variable_scope,
                             ScopedObject,
                             deprecated)
from .defaults import get_option_defaults, OptionDefaults

__all__ = ['Model']


class Model(ScopedObject):
    """Scaffold for defining models with TensorFlow.

    This class provides a basic scaffold for defining models with TensorFlow.
    It settles a common practice for organizing the variables of parameters,
    as well as variables for training and prediction

    Furthermore, it also provides a set of methods to export and import model
    parameters as numpy arrays.  It might be useful for reusing part of the
    model parameters in a customized way.

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

    #: default options of this model
    OPTION_DEFAULTS = None      # type: OptionDefaults

    def __init__(self, name=None, default_name=None, global_step=None):
        super(Model, self).__init__(name=name, default_name=default_name)

        # whether or not the model has been built?
        self._has_built = False

        # the global step variable of this model.
        self._global_step = global_step     # type: tf.Variable

        # the placeholders and operations for set parameter values
        self._set_param_value_ph_op = {}
        self._set_all_param_values_op = None

        # memorize the default parameter values at construction time
        option_defaults = get_option_defaults()
        if self.OPTION_DEFAULTS:
            option_defaults = self.OPTION_DEFAULTS.merge(option_defaults)
        self._option_defaults_at_construction = option_defaults
        self._cached_option_defaults = [None, None]

    @property
    def option_defaults(self):
        """Get the default options.
         
        The returned default parameters would be merged from current 
        active context and from those specified at construction time.

        Returns
        -------
        OptionDefaults
            The default parameters.
        """
        defaults = get_option_defaults()
        if defaults is self._option_defaults_at_construction:
            return defaults
        if defaults is not self._cached_option_defaults[0]:
            self._cached_option_defaults[1] = \
                defaults.merge(self._option_defaults_at_construction)
            self._cached_option_defaults[0] = defaults
        return self._cached_option_defaults[1]

    @property
    @deprecated
    def model_variable_scope(self):
        return self.variable_scope

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

        with open_variable_scope(self._variable_scope):
            # create the global step variable if there's none
            if self._global_step is None:
                self._global_step = tf.get_variable(
                    'global_step', dtype=tf.int64, trainable=False,
                    initializer=np.asarray(0, dtype=np.int64)
                )

            # build the model objects
            self._build()

            # build the placeholders and operation to set parameter values
            with tf.name_scope('set_param_values'):
                for name, var in six.iteritems(self.get_param_variables()):
                    ph_dtype = var.dtype.base_dtype
                    ph_name = 'assign_ph'
                    op_name = 'assign_op'
                    ph = tf.placeholder(ph_dtype, var.get_shape(), ph_name)
                    op = tf.assign(var, ph, name=op_name)
                    self._set_param_value_ph_op[name] = (ph, op)
                self._set_all_param_values_op = tf.group(*(
                    v[1] for v in six.itervalues(self._set_param_value_ph_op)
                ))

    def _build(self):
        """Build the model.

        Derived class should override this method to actually establish
        all TensorFlow graph nodes for training and evaluating this model.

        All the variables defined in this method will be added to `scope`,
        so they can be retrieved by `tf.get_collection` according to this
        scope.  However, for model parameter variables, they should also
        be added to `tf.GraphKeys.MODEL_VARIABLES` collection, in order to
        be distinguished from auxiliary variables used for training or
        evaluation.
        
        Note that the derived classes should use the parameter values defined
        in `param_defaults`, so that the behavior of `build()` can follow
        the context at construction time.
        """
        raise NotImplementedError()

    def get_model_variables(self, collection=tf.GraphKeys.GLOBAL_VARIABLES):
        """Get the model variables.

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
        scope_name = self.variable_scope.name + '/'
        scope_name_len = len(scope_name)
        return {
            var.name[scope_name_len:].rsplit(':', 1)[0]: var
            for var in tf.get_collection(collection, scope_name)
        }

    def get_param_variables(self, trainable=None):
        """Get the model parameter variables.

        The parameter variables are the variables in `_variable_scope`
        and `tf.GraphKeys.MODEL_VARIABLES` collection.

        Although it is possible to override this method, so as to include
        more variables other than those specified in `MODEL_VARIABLES`
        collection, it is generally not a good practice.

        Parameters
        ----------
        trainable : bool
            If set to True, will only get the trainable parameters.

        Returns
        -------
        dict[str, tf.Variable]
            Dict which maps from relative names to variable objects.
        """
        model_vars = self.get_model_variables(tf.GraphKeys.MODEL_VARIABLES)
        if trainable:
            collection = tf.GraphKeys.TRAINABLE_VARIABLES
            train_vars = self.get_model_variables(collection)
            model_vars = {k: v for k, v in six.iteritems(model_vars)
                          if k in train_vars}
        return model_vars

    def get_param_values(self):
        """Get the model parameter values as numpy arrays.

        Returns
        -------
        dict[str, np.ndarray]
            Dict which maps from relative names to parameter values.
        """
        session = get_default_session_or_error()
        param_vars = self.get_param_variables()
        param_names = list(param_vars.keys())
        param_values = session.run([param_vars[n] for n in param_names])
        return {n: v for n, v in zip(param_names, param_values)}

    def set_param_values(self, param_values, partial_set=False):
        """Set the model parameter values as numpy arrays.

        Parameters
        ----------
        param_values : dict[str, np.ndarray]
            The parameter values, map from relative names to numpy arrays.

        partial_set : bool
            If not set to True, a KeyError will be raised if any of the
            model parameter variables are not specified in `param_values`.
        """
        session = get_default_session_or_error()

        # check the specified values
        param_vars = self.get_param_variables()
        param_names = list(param_vars.keys())
        all_variables_set = True
        for n in param_names:
            if n not in param_values:
                if not partial_set:
                    raise KeyError(
                        'Value of parameter %r is not specified.' % (n,))
                else:
                    all_variables_set = False
                    break

        # assign values to variables
        if all_variables_set:
            feed_dict = {
                self._set_param_value_ph_op[n][0]: param_values[n]
                for n in param_names
            }
            session.run(self._set_all_param_values_op, feed_dict=feed_dict)
        else:
            for n, value in six.iteritems(param_values):
                if n in param_vars:
                    ph, op = self._set_param_value_ph_op[n]
                    session.run(op, feed_dict={ph: value})

    def ensure_variables_initialized(self):
        """Initialize all uninitialized model variables.
        
        If the `global_step` is created by the model itself, then it will
        also be initialized.  Otherwise the `global_step` must be manually
        initialized.
        """
        var_list = list(six.itervalues(self.get_model_variables()))
        ensure_variables_initialized(var_list)
