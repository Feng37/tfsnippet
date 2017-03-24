# -*- coding: utf-8 -*-
import numpy as np
import six
import tensorflow as tf

from tfsnippet.utils import camel_to_underscore, get_default_session_or_error

__all__ = ['Model']


class Model(object):
    """Scaffold for defining models with TensorFlow.

    This class provides a basic scaffold for defining models with TensorFlow.
    It settles a common practice for organizing the variables of parameters,
    as well as training and prediction variables.

    Furthermore, it also provides a set of methods to export and import model
    parameters as numpy arrays.  It might be useful for reusing part of the
    model parameters in a customized way.

    Parameters
    ----------
    name : str
        Name of the model.
    """

    def __init__(self, name=None):
        self.name = name

        # whether or not the model has been built?
        self._has_built = False

        # the variable scope of all variables defined in this model.
        self._model_varscope = None     # type: tf.VariableScope

        # the placeholders and operations for set parameter values
        self._set_param_value_ph_op = {}
        self._set_all_param_values_op = None

    @property
    def model_variable_scope(self):
        """Get the model variable scope."""
        self.build()
        return self._model_varscope

    @property
    def has_built(self):
        """Whether or not the model has been built?"""
        return self._has_built

    def build(self):
        """Build the model and the trainer.

        Although this method will be called automatically when the model
        is required to be built, however, it is recommended to call this
        method soon after the model object is constructed.
        Otherwise the namespace of variables defined in this model might
        not match the scope where the model object is constructed.
        """
        if self._has_built:
            return
        self._has_built = True

        default_name = camel_to_underscore(self.__class__.__name__)
        with tf.variable_scope(self.name, default_name=default_name) as scope:
            # store the model variable scope
            self._model_varscope = scope

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
        """
        raise NotImplementedError()

    def get_model_variables(self, collection=tf.GraphKeys.GLOBAL_VARIABLES):
        """Get the model variables.

        This method will get variables which are in `model_variable_scope`
        and the specified `collection`, as a dict which maps relative
        names to variable objects.  By "relative name" we mean to remove
        the name of `model_variable_scope` from the front of variable names.

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
        scope_name = self.model_variable_scope.name + '/'
        scope_name_len = len(scope_name)
        return {
            var.name[scope_name_len:].rsplit(':', 1)[0]: var
            for var in tf.get_collection(collection, scope_name)
        }

    def get_param_variables(self):
        """Get the model parameter variables.

        The parameter variables are the variables in `model_variable_scope`
        and `tf.GraphKeys.MODEL_VARIABLES` collection.

        Although it is possible to override this method, so as to include
        more variables other than those specified in `MODEL_VARIABLES`
        collection, it is generally not a good practice.

        Returns
        -------
        dict[str, tf.Variable]
            Dict which maps from relative names to variable objects.
        """
        return self.get_model_variables(tf.GraphKeys.MODEL_VARIABLES)

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
