# -*- coding: utf-8 -*-
import os
import unittest

import tensorflow as tf

from tfsnippet.scaffold import Model
from tfsnippet.utils import (TemporaryDirectory, set_variable_values,
                             get_variable_values, get_uninitialized_variables)
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

    def test_construction(self):
        with tf.Graph().as_default():
            # the variable out of model scope
            _ = tf.get_variable('out_var', shape=(), dtype=tf.int32)

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

            # test name de-duplication
            self.assertEqual(_MyModel().variable_scope.name, 'my_model_1')
            self.assertEqual(_MyModel('the_model').variable_scope.name,
                             'the_model')

    def test_set_global_step(self):
        with tf.Graph().as_default():
            model = _MyModel()
            self.assertEqual(model.get_global_step().name,
                             'my_model/global_step:0')

        with tf.Graph().as_default():
            model = _MyModel()
            model.set_global_step(
                tf.get_variable('global_step', dtype=tf.int32, initializer=0))
            self.assertEqual(model.get_global_step().name, 'global_step:0')

        with tf.Graph().as_default():
            model = _MyModel()
            model.build()
            with self.assertRaisesRegex(
                    RuntimeError, 'The `global_step` variable has already been '
                                  'set or created.'):
                model.set_global_step(
                    tf.get_variable('global_step', dtype=tf.int32,
                                    initializer=0)
                )

    def test_initialize_variables(self):
        with tf.Graph().as_default(), tf.Session():
            model = _MyModel()
            model.build()
            out_var = tf.get_variable('out_var', shape=(), dtype=tf.int32)

            # test initializing variables
            self.assertEqual(
                set(get_uninitialized_variables()),
                {model.model_var, model.nested_var, model.other_var,
                 model.get_global_step(), out_var}
            )
            model.ensure_variables_initialized()
            self.assertEqual(
                set(get_uninitialized_variables()),
                {out_var}
            )

    def test_load_save(self):
        with tf.Graph().as_default(), tf.Session():
            model = _MyModel()
            model.ensure_variables_initialized()

            with TemporaryDirectory() as tempdir:
                # test save and load model
                model.save_model(tempdir)
                self.assertTrue(os.path.isfile(os.path.join(tempdir, 'latest')))

                # test overwriting directory
                model.save_model(os.path.join(tempdir, '1'))
                with self.assertRaisesRegex(IOError, '.*already exists.'):
                    model.save_model(os.path.join(tempdir, '1'))
                model.save_model(os.path.join(tempdir, '1'), overwrite=True)
                self.assertTrue(
                    os.path.isfile(os.path.join(tempdir, '1', 'latest')))

                # test overwriting file
                open(os.path.join(tempdir, '2'), 'wb').close()
                with self.assertRaisesRegex(IOError, '.*already exists.'):
                    model.save_model(os.path.join(tempdir, '2'))
                model.save_model(os.path.join(tempdir, '2'), overwrite=True)
                self.assertTrue(
                    os.path.isfile(os.path.join(tempdir, '2', 'latest')))

                # test load from tempdir
                self.assertEqual(
                    get_variable_values([model.model_var, model.nested_var]),
                    [1, 3]
                )
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
                    model.load_model(os.path.join(tempdir, '3'))

if __name__ == '__main__':
    unittest.main()
