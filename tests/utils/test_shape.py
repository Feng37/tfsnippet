# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.utils import (get_dimension_size,
                             is_deterministic_shape,
                             get_dynamic_tensor_shape,
                             maybe_explicit_broadcast)
from tests.helper import TestCase


class ShapeTestCase(TestCase):

    def test_get_dimension_size(self):
        with self.get_session():
            x = tf.placeholder(tf.float32, shape=[None, 3])
            dim = tf.placeholder(tf.int32, shape=())
            shape = tf.placeholder(tf.int32, shape=[None])
            x_data = np.arange(36).reshape([-1, 3])

            # test get static shape
            self.assertEqual(
                get_dimension_size(x, 1),
                3
            )
            self.assertEqual(
                get_dimension_size(x, -1),
                3
            )

            # test get dynamic shape
            self.assertEqual(
                get_dimension_size(x, 0).eval({x: x_data}),
                12
            )
            self.assertEqual(
                get_dimension_size(x, dim).eval({x: x_data, dim: 0}),
                12
            )
            self.assertEqual(
                get_dimension_size(x, dim).eval({x: x_data, dim: 1}),
                3
            )

            # test get the dimension of tensor with even unknown number of
            # dimensions
            x_reshaped = tf.reshape(x, shape)
            self.assertEqual(
                get_dimension_size(x_reshaped, 0).eval({
                    x: x_data,
                    shape: [3, 6, 2]
                }),
                3
            )
            self.assertEqual(
                get_dimension_size(x_reshaped, dim).eval({
                    x: x_data,
                    dim: 1,
                    shape: [3, 6, 2]
                }),
                6
            )
            self.assertEqual(
                get_dimension_size(x_reshaped, dim).eval({
                    x: x_data,
                    dim: 2,
                    shape: [3, 6, 2]
                }),
                2
            )

    def test_get_dynamic_tensor_shape(self):
        with self.get_session():
            x = tf.placeholder(tf.float32, shape=[None, 3, 4])
            shape = tf.placeholder(tf.int32, shape=[None])
            x_data = np.arange(24).reshape([-1, 3, 4])

            # test get static shape
            self.assertEqual(get_dynamic_tensor_shape(x[0, ...]), (3, 4))
            self.assertEqual(
                get_dynamic_tensor_shape(x, lambda s: s[1:]),
                (3, 4)
            )

            # test get dynamic shape
            x_shape = get_dynamic_tensor_shape(x)
            self.assertIsInstance(x_shape, tf.Tensor)
            self.assertEqual(x_shape.get_shape().as_list(), [3])
            self.assertEqual(tuple(x_shape.eval({x: x_data})), (2, 3, 4))

            x_shape = get_dynamic_tensor_shape(x[:, :, 0])
            self.assertIsInstance(x_shape, tf.Tensor)
            self.assertEqual(x_shape.get_shape().as_list(), [2])
            self.assertEqual(tuple(x_shape.eval({x: x_data})), (2, 3))

            x_shape = get_dynamic_tensor_shape(x, lambda s: s[:-1])
            self.assertIsInstance(x_shape, tf.Tensor)
            self.assertEqual(x_shape.get_shape().as_list(), [2])
            self.assertEqual(tuple(x_shape.eval({x: x_data})), (2, 3))

            # test get the shape of tensor with unknown number of dimensions
            x_reshaped = tf.reshape(x, shape)
            x_shape = get_dynamic_tensor_shape(x_reshaped)
            self.assertIsInstance(x_shape, tf.Tensor)
            self.assertEqual(x_shape.get_shape().as_list(), [None])
            self.assertEqual(
                tuple(x_shape.eval({x: x_data, shape: [2, 3, 2, 2]})),
                (2, 3, 2, 2)
            )

            x_shape = get_dynamic_tensor_shape(x_reshaped[0, ...])
            self.assertIsInstance(x_shape, tf.Tensor)
            self.assertEqual(x_shape.get_shape().as_list(), [None])
            self.assertEqual(
                tuple(x_shape.eval({x: x_data, shape: [2, 3, 2, 2]})),
                (3, 2, 2)
            )

            x_shape = get_dynamic_tensor_shape(x_reshaped, lambda s: s[: -1])
            self.assertIsInstance(x_shape, tf.Tensor)
            self.assertEqual(x_shape.get_shape().as_list(), [None])
            self.assertEqual(
                tuple(x_shape.eval({x: x_data, shape: [2, 3, 2, 2]})),
                (2, 3, 2)
            )

    def test_is_deterministic_shape(self):
        deterministic_shape = [
            (),
            (1,),
            (1, 2),
            [],
            [1],
            [1, 2],
            [np.asarray([1], dtype=np.int)[0]],
            tf.placeholder(tf.float32, shape=(1, 2)).get_shape(),
        ]
        for v in deterministic_shape:
            self.assertTrue(
                is_deterministic_shape(v),
                msg='%r should be a deterministic shape.' % (v,)
            )
        non_deterministic_shape = [
            tf.placeholder(tf.int32, shape=(3,)),
            tf.placeholder(tf.int32, shape=(None,)),
            [tf.placeholder(tf.int32, shape=()), 1],
            tf.placeholder(tf.float32, shape=(1, None)).get_shape(),
            tf.get_variable('shape', shape=(1,), dtype=tf.int32),
            tf.placeholder(tf.int32),
            [tf.placeholder(tf.int32), 1, tf.placeholder(tf.int32, ())],
        ]
        for v in non_deterministic_shape:
            self.assertFalse(
                is_deterministic_shape(v),
                msg='%r should not be a deterministic shape.' % (v,)
            )
        not_shape_type = [
            object(),
            {},
            '',
            True,
            1,
            1.0,
            [True],
            [[1, 2]],
            [1.0],
            tf.placeholder(tf.float32, shape=(1,)),
            tf.placeholder(tf.int32, shape=()),
            tf.placeholder(tf.int32, shape=(1, 2)),
            [tf.placeholder(tf.int32, shape=(1,))],
            [tf.placeholder(tf.float32, shape=())],
            tf.placeholder(tf.float32),
            [tf.placeholder(tf.float32), 1, tf.placeholder(tf.float32, ())],
        ]
        for v in not_shape_type:
            with self.assertRaises(TypeError,
                                   msg='%r is not a shape.' % (v,)):
                is_deterministic_shape(v)

    def test_maybe_explicit_broadcast(self):
        with self.get_session() as session:
            # test on equal static shape
            x = tf.convert_to_tensor(np.arange(2))
            y = tf.convert_to_tensor(np.arange(2, 4))
            xx, yy = maybe_explicit_broadcast(x, y)
            self.assertIs(xx, x)
            self.assertEqual(xx.get_shape().as_list(), [2])
            np.testing.assert_equal(xx.eval(), [0, 1])
            self.assertIs(yy, y)
            self.assertEqual(yy.get_shape().as_list(), [2])
            np.testing.assert_equal(yy.eval(), [2, 3])

            # test on fully static same dimensional shape
            x = tf.convert_to_tensor(np.arange(2).reshape([1, 2]))
            y = tf.convert_to_tensor(np.arange(2, 4).reshape([2, 1]))
            xx, yy = maybe_explicit_broadcast(x, y)
            self.assertEqual(xx.get_shape().as_list(), [2, 2])
            np.testing.assert_equal(xx.eval(), [[0, 1], [0, 1]])
            self.assertEqual(yy.get_shape().as_list(), [2, 2])
            np.testing.assert_equal(yy.eval(), [[2, 2], [3, 3]])

            # test on fully static different dimensional shape
            x = tf.convert_to_tensor(np.arange(2).reshape([2, 1]))
            y = tf.convert_to_tensor(np.arange(2, 4))
            xx, yy = maybe_explicit_broadcast(x, y)
            self.assertEqual(xx.get_shape().as_list(), [2, 2])
            np.testing.assert_equal(xx.eval(), [[0, 0], [1, 1]])
            self.assertEqual(yy.get_shape().as_list(), [2, 2])
            np.testing.assert_equal(yy.eval(), [[2, 3], [2, 3]])

            # test on dynamic same dimensional shape
            x = tf.placeholder(tf.float32, (None, 3))
            y = tf.placeholder(tf.float32, (2, None))
            xx, yy = maybe_explicit_broadcast(x, y)
            x_data = np.arange(3).reshape((1, 3))
            y_data = np.arange(3, 5).reshape((2, 1))
            self.assertEqual(xx.get_shape().as_list(), [2, 3])
            np.testing.assert_equal(xx.eval({x: x_data, y: y_data}),
                                    [[0, 1, 2], [0, 1, 2]])
            self.assertEqual(yy.get_shape().as_list(), [2, 3])
            np.testing.assert_equal(yy.eval({x: x_data, y: y_data}),
                                    [[3, 3, 3], [4, 4, 4]])

            # test on dynamic different dimensional shape
            x = tf.placeholder(tf.float32, (None,))
            y = tf.placeholder(tf.float32, (None, 2))
            xx, yy = maybe_explicit_broadcast(x, y)
            x_data = np.arange(1, 2)
            y_data = np.arange(2, 6).reshape((2, 2))
            self.assertEqual(xx.get_shape().as_list(), [None, 2])
            np.testing.assert_equal(xx.eval({x: x_data, y: y_data}),
                                    [[1, 1], [1, 1]])
            self.assertEqual(yy.get_shape().as_list(), [None, 2])
            np.testing.assert_equal(yy.eval({x: x_data, y: y_data}),
                                    [[2, 3], [4, 5]])

            # test `tail_no_broadcast_ndims = 0`
            x = tf.convert_to_tensor(np.arange(2).reshape([1, 1, 2]))
            y = tf.convert_to_tensor(np.arange(2, 4).reshape([2, 1]))
            xx, yy = maybe_explicit_broadcast(x, y, tail_no_broadcast_ndims=0)
            self.assertEqual(xx.get_shape().as_list(), [1, 2, 2])
            np.testing.assert_equal(xx.eval(), [[[0, 1], [0, 1]]])
            self.assertEqual(yy.get_shape().as_list(), [1, 2, 2])
            np.testing.assert_equal(yy.eval(), [[[2, 2], [3, 3]]])

            # test `tail_no_broadcast_ndims = 1`
            x = tf.convert_to_tensor(np.arange(2).reshape([1, 1, 2]))
            y = tf.convert_to_tensor(np.arange(2, 4).reshape([2, 1]))
            xx, yy = maybe_explicit_broadcast(x, y, tail_no_broadcast_ndims=1)
            self.assertEqual(xx.get_shape().as_list(), [1, 2, 2])
            np.testing.assert_equal(xx.eval(), [[[0, 1], [0, 1]]])
            self.assertEqual(yy.get_shape().as_list(), [1, 2, 1])
            np.testing.assert_equal(yy.eval(), [[[2], [3]]])

            # test `tail_no_broadcast_ndims = 2`
            x = tf.convert_to_tensor(np.arange(2).reshape([1, 1, 2]))
            y = tf.convert_to_tensor(np.arange(2, 4).reshape([2, 1]))
            xx, yy = maybe_explicit_broadcast(x, y, tail_no_broadcast_ndims=2)
            self.assertEqual(xx.get_shape().as_list(), [1, 1, 2])
            np.testing.assert_equal(xx.eval(), [[[0, 1]]])
            self.assertEqual(yy.get_shape().as_list(), [1, 2, 1])
            np.testing.assert_equal(yy.eval(), [[[2], [3]]])

            # test `tail_no_broadcast_ndims = 1` on dynamic shape
            x = tf.placeholder(tf.float32, (None, None, None))
            y = tf.placeholder(tf.float32, (None, None))
            xx, yy = maybe_explicit_broadcast(x, y, tail_no_broadcast_ndims=1)
            x_data = np.arange(2).reshape([1, 1, 2])
            y_data = np.arange(2, 4).reshape([2, 1])
            self.assertEqual(xx.get_shape().as_list(), [None, None, None])
            np.testing.assert_equal(xx.eval({x: x_data, y: y_data}),
                                    [[[0, 1], [0, 1]]])
            self.assertEqual(yy.get_shape().as_list(), [None, None, None])
            np.testing.assert_equal(yy.eval({x: x_data, y: y_data}),
                                    [[[2], [3]]])

            # test `tail_no_broadcast_ndims = 1` on fully dynamic shape
            x = tf.placeholder(tf.float32)
            y = tf.placeholder(tf.float32)
            xx, yy = maybe_explicit_broadcast(x, y, tail_no_broadcast_ndims=1)
            x_data = np.arange(2).reshape([1, 1, 2])
            y_data = np.arange(2, 4).reshape([2, 1])
            self.assertEqual(xx.get_shape().ndims, None)
            np.testing.assert_equal(xx.eval({x: x_data, y: y_data}),
                                    [[[0, 1], [0, 1]]])
            self.assertEqual(yy.get_shape().ndims, None)
            np.testing.assert_equal(yy.eval({x: x_data, y: y_data}),
                                    [[[2], [3]]])

            # test error on static shape
            with self.assertRaisesRegex(
                    ValueError, '.* and .* cannot broadcast to match.*'):
                _ = maybe_explicit_broadcast(
                    tf.convert_to_tensor(np.arange(2)),
                    tf.convert_to_tensor(np.arange(3))
                )

            # test error on dynamic shape
            x = tf.placeholder(tf.float32, (None, 3))
            y = tf.placeholder(tf.float32, (2, None))
            xx, yy = maybe_explicit_broadcast(x, y)
            x_data = np.arange(3).reshape((1, 3))
            y_data = np.arange(3, 7).reshape((2, 2))
            with self.assertRaisesRegex(Exception, r'.*Incompatible shapes.*'):
                session.run([xx, yy], feed_dict={x: x_data, y: y_data})

if __name__ == '__main__':
    unittest.main()
