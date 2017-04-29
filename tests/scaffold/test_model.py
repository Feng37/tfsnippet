# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf

from tfsnippet.scaffold import Model


class MyModel(Model):

    def _build(self):
        self.model_var = tf.get_variable(
            'model_var',
            initializer=1,
            dtype=tf.int32,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                         tf.GraphKeys.MODEL_VARIABLES]
        )
        self.other_var = tf.get_variable(
            'other_var',
            initializer=2,
            dtype=tf.int32,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES]
        )
        with tf.variable_scope('nested'):
            self.nested_var = tf.get_variable(
                'nested_var',
                initializer=3,
                dtype=tf.int32,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                             tf.GraphKeys.MODEL_VARIABLES]
            )


class ModelTestCase(unittest.TestCase):

    def test_basic(self):
        with tf.Graph().as_default():
            # the variable out of model scope
            out_var = tf.get_variable(
                'out_var',
                initializer=-1,
                dtype=tf.int32,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                             tf.GraphKeys.MODEL_VARIABLES]
            )

            # test build
            model = MyModel()
            self.assertFalse(model.has_built)
            model.build()
            self.assertTrue(model.has_built)
            self.assertEqual(model.variable_scope.name, 'my_model')

            # test get all variables
            self.assertEqual(
                model.get_variables(),
                {
                    'model_var': model.model_var,
                    'other_var': model.other_var,
                    'nested/nested_var': model.nested_var,
                    'global_step': model.get_global_step()
                }
            )

            # test get parameter variables
            self.assertEqual(
                model.get_param_variables(),
                {'model_var': model.model_var,
                 'nested/nested_var': model.nested_var}
            )

            # test get parameter values and set parameter values
            init_op = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init_op)
                self.assertEqual(
                    model.get_param_values(),
                    {'model_var': 1, 'nested/nested_var': 3}
                )
                model.set_param_values(
                    {'model_var': 10, 'nested/nested_var': 30})
                self.assertEqual(
                    model.get_param_values(),
                    {'model_var': 10, 'nested/nested_var': 30}
                )
                with self.assertRaises(KeyError):
                    model.set_param_values({'model_var': 10})
                model.set_param_values(
                    {'model_var': 100}, partial_set=True)
                self.assertEqual(
                    model.get_param_values(),
                    {'model_var': 100, 'nested/nested_var': 30}
                )

            # test name de-duplication
            self.assertEqual(MyModel().variable_scope.name, 'my_model_1')
            self.assertEqual(MyModel('the_model').variable_scope.name,
                             'the_model')
