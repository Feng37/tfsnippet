# -*- coding: utf-8 -*-
import os
import unittest

import tensorflow as tf

from tfsnippet.scaffold import Model
from tfsnippet.utils import (TemporaryDirectory, set_variable_values,
                             get_variable_values)
from tests.helper import TestCase


class _MyModel(Model):

    def _build(self):
        with tf.variable_scope('model'):
            self.model_var = tf.get_variable(
                'model_var',
                initializer=1,
                dtype=tf.int32
            )
            with tf.variable_scope('nested'):
                self.nested_var = tf.get_variable(
                    'nested_var',
                    initializer=3,
                    dtype=tf.int32,
                    trainable=False
                )
        self.other_var = tf.get_variable(
            'other_var',
            initializer=2,
            dtype=tf.int32
        )


class ModelTestCase(TestCase):

    def test_Model(self):
        with tf.Graph().as_default():
            # the variable out of model scope
            out_var = tf.get_variable(
                'out_var',
                initializer=-1,
                dtype=tf.int32
            )

            # test build
            model = _MyModel()
            self.assertFalse(model.has_built)
            model.build()
            self.assertTrue(model.has_built)
            self.assertEqual(model.variable_scope.name, 'my_model')

            # test get all variables
            self.assertEqual(
                model.get_variables(),
                {
                    'model/model_var': model.model_var,
                    'model/nested/nested_var': model.nested_var,
                    'other_var': model.other_var,
                    'global_step': model.get_global_step()
                }
            )

            # test get parameter variables
            self.assertEqual(
                model.get_param_variables(),
                {'model/model_var': model.model_var,
                 'model/nested/nested_var': model.nested_var}
            )
            self.assertEqual(
                model.get_param_variables(tf.GraphKeys.TRAINABLE_VARIABLES),
                {'model/model_var': model.model_var}
            )

            # test save and load model
            init_op = tf.global_variables_initializer()
            with tf.Session() as session, TemporaryDirectory() as tempdir:
                session.run(init_op)
                model.save_model(tempdir)
                model.save_model(os.path.join(tempdir, '1'))

                with self.assertRaisesRegex(IOError, '.*already exists.'):
                    model.save_model(tempdir)

                self.assertEqual(
                    get_variable_values([model.model_var, model.nested_var]),
                    [1, 3]
                )

                # test load from tempdir
                set_variable_values(
                    [model.model_var, model.nested_var],
                    [10, 30]
                )
                self.assertEqual(
                    get_variable_values([model.model_var, model.nested_var]),
                    [10, 30]
                )
                model.load_model(tempdir)
                self.assertEqual(
                    get_variable_values([model.model_var, model.nested_var]),
                    [1, 3]
                )

                # test load from tempdir + '/1'
                set_variable_values(
                    [model.model_var, model.nested_var],
                    [10, 30]
                )
                self.assertEqual(
                    get_variable_values([model.model_var, model.nested_var]),
                    [10, 30]
                )
                model.load_model(os.path.join(tempdir, '1'))
                self.assertEqual(
                    get_variable_values([model.model_var, model.nested_var]),
                    [1, 3]
                )

                # test load from non-exist
                with self.assertRaisesRegex(
                        IOError, 'Checkpoint file does not exist.*'):
                    model.load_model(os.path.join(tempdir, '2'))

            # test name de-duplication
            self.assertEqual(_MyModel().variable_scope.name, 'my_model_1')
            self.assertEqual(_MyModel('the_model').variable_scope.name,
                             'the_model')


if __name__ == '__main__':
    unittest.main()
