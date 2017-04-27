# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.utils import (get_dimension_size,
                             is_deterministic_shape,
                             repeat_tensor_for_samples,
                             ReshapeHelper)


class ShapeTestCase(unittest.TestCase):

    def test_get_dimension_size(self):
        with tf.Graph().as_default(), tf.Session().as_default():
            x = tf.placeholder(tf.float32, shape=[None, 3])
            dim = tf.placeholder(tf.int32, shape=())
            shape = tf.placeholder(tf.int32, shape=[None])
            x_data = np.arange(36).reshape([-1, 3])

            # test get static shape
            self.assertEqual(
                get_dimension_size(x, 1),
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

            # test out of range
            with self.assertRaises(IndexError):
                get_dimension_size(x, -1)
            with self.assertRaises(IndexError):
                get_dimension_size(x, 2)

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

    def test_is_deterministic_shape(self):
        with tf.Graph().as_default():
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
                [tf.placeholder(tf.float32, shape=())]
            ]
            for v in not_shape_type:
                with self.assertRaises(TypeError,
                                       msg='%r is not a shape.' % (v,)):
                    is_deterministic_shape(v)


    def test_repeat_tensor_for_samples(self):
        def check_function(x_data, sz_data, bz_data):
            return np.tile(
                x_data,
                [sz_data * bz_data // x_data.shape[0]] +
                [1] * (len(x_data.shape) - 1)
            )

        with tf.Graph().as_default(), tf.Session().as_default():
            x = tf.placeholder(tf.float32, (None, 3, 4))
            sample_size = tf.placeholder(tf.int32, ())
            batch_size = tf.placeholder(tf.int32, ())
            x_data = np.arange(48).reshape([-1, 3, 4])
            sz_data = 3
            bz_data = 4
            self.assertTrue(
                np.all(
                    repeat_tensor_for_samples(
                        x, sample_size, batch_size
                    ).eval({
                        x: x_data, sample_size: sz_data, batch_size: bz_data
                    }) ==
                    check_function(x_data, sz_data, bz_data)
                )
            )



    def test_ReshapeHelper(self):
        with tf.Graph().as_default(), tf.Session().as_default():
            helper = ReshapeHelper()

            # test an empty reshape helper
            self.assertTrue(helper.is_deterministic)
            self.assertEqual(helper.get_static_shape().as_list(),
                             [])
            self.assertEqual(helper.get_dynamic_shape(),
                             ())

            # test add an integer as dimension
            helper.add(1)
            self.assertTrue(helper.is_deterministic)
            self.assertEqual(helper.get_static_shape().as_list(),
                             [1])
            self.assertEqual(helper.get_dynamic_shape(),
                             (1,))

            # test add a series of integers
            helper.add([2, 3])
            self.assertTrue(helper.is_deterministic)
            self.assertEqual(helper.get_static_shape().as_list(),
                             [1, 2, 3])
            self.assertEqual(helper.get_dynamic_shape(),
                             (1, 2, 3))

            # test add a deterministic tensor shape
            x = tf.placeholder(dtype=tf.int32, shape=[None, 4, 5, 6])
            helper.add_template(x, lambda v: v[1: 3])
            self.assertTrue(helper.is_deterministic)
            self.assertEqual(helper.get_static_shape().as_list(),
                             [1, 2, 3, 4, 5])
            self.assertEqual(helper.get_dynamic_shape(),
                             (1, 2, 3, 4, 5))

            # test add non-deterministic tensor shape, which should cause error
            for payload in [x.get_shape(), x.get_shape()[0]]:
                with self.assertRaises(ValueError) as cm:
                    helper.add(payload)
                self.assertIn('You should use `add_template`',
                              str(cm.exception))

            # test add `-1`
            helper.add(-1)
            self.assertTrue(helper.is_deterministic)
            self.assertEqual(helper.get_static_shape().as_list(),
                             [1, 2, 3, 4, 5, None])
            self.assertEqual(helper.get_dynamic_shape(),
                             (1, 2, 3, 4, 5, -1))

            # test add `-1` for the 2nd time, which should cause error
            with self.assertRaises(ValueError) as cm:
                helper.add(-1)
            self.assertIn('"-1" can only appear for at most once',
                          str(cm.exception))
            with self.assertRaises(ValueError) as cm:
                helper.add([1, -1, 2])
            self.assertIn('"-1" can only appear for at most once',
                          str(cm.exception))

            # test add tensor as dimension
            dim_1 = tf.placeholder(dtype=tf.int32, shape=())
            helper.add(dim_1)
            self.assertFalse(helper.is_deterministic)
            self.assertEqual(helper.get_static_shape().as_list(),
                             [1, 2, 3, 4, 5, None, None])
            self.assertIsInstance(helper.get_dynamic_shape(),
                                  tf.Tensor)
            self.assertEqual(
                tuple(helper.get_dynamic_shape().eval({dim_1: 6})),
                (1, 2, 3, 4, 5, -1, 6)
            )

            # test add tensor as shape
            shape_1 = tf.placeholder(dtype=tf.int32, shape=(2,))
            helper.add(shape_1)
            self.assertFalse(helper.is_deterministic)
            self.assertEqual(helper.get_static_shape().as_list(),
                             [1, 2, 3, 4, 5, None, None, None, None])
            self.assertIsInstance(helper.get_dynamic_shape(),
                                  tf.Tensor)
            self.assertEqual(
                tuple(helper.get_dynamic_shape().eval({
                    dim_1: 6, shape_1: [7, 8]
                })),
                (1, 2, 3, 4, 5, -1, 6, 7, 8)
            )

            # test to add a tuple of dimensions
            dim_2 = tf.placeholder(dtype=tf.int32, shape=())
            helper.add([dim_1, dim_2, 11])
            self.assertFalse(helper.is_deterministic)
            self.assertEqual(
                helper.get_static_shape().as_list(),
                [1, 2, 3, 4, 5, None, None, None, None, None, None, 11]
            )
            self.assertIsInstance(helper.get_dynamic_shape(),
                                  tf.Tensor)
            self.assertEqual(
                tuple(helper.get_dynamic_shape().eval({
                    dim_1: 6,
                    dim_2: 10,
                    shape_1: [7, 8],
                })),
                (1, 2, 3, 4, 5, -1, 6, 7, 8, 6, 10, 11)
            )

            # test add partial shape of `x`
            helper.add_template(x, lambda s: s[: 2])
            self.assertFalse(helper.is_deterministic)
            self.assertEqual(
                helper.get_static_shape().as_list(),
                [1, 2, 3, 4, 5, None, None, None, None, None, None, 11,
                 None, 4]
            )
            self.assertIsInstance(helper.get_dynamic_shape(),
                                  tf.Tensor)
            self.assertEqual(
                tuple(helper.get_dynamic_shape().eval({
                    dim_1: 6,
                    dim_2: 10,
                    shape_1: [7, 8],
                    x: np.arange(120 * 9).reshape([-1, 4, 5, 6])
                })),
                (1, 2, 3, 4, 5, -1, 6, 7, 8, 6, 10, 11, 9, 4)
            )

            # test add dimension of `x`
            helper.add_template(x, lambda s: s[0])
            self.assertFalse(helper.is_deterministic)
            self.assertEqual(
                helper.get_static_shape().as_list(),
                [1, 2, 3, 4, 5, None, None, None, None, None, None, 11,
                 None, 4, None]
            )
            self.assertIsInstance(helper.get_dynamic_shape(),
                                  tf.Tensor)
            self.assertEqual(
                tuple(helper.get_dynamic_shape().eval({
                    dim_1: 6,
                    dim_2: 10,
                    shape_1: [7, 8],
                    x: np.arange(120 * 9).reshape([-1, 4, 5, 6])
                })),
                (1, 2, 3, 4, 5, -1, 6, 7, 8, 6, 10, 11, 9, 4, 9)
            )

            # test add deterministic dimension of `x`
            helper.add_template(x, lambda s: s[1])
            self.assertFalse(helper.is_deterministic)
            self.assertEqual(
                helper.get_static_shape().as_list(),
                [1, 2, 3, 4, 5, None, None, None, None, None, None, 11,
                 None, 4, None, 4]
            )
            self.assertIsInstance(helper.get_dynamic_shape(),
                                  tf.Tensor)
            self.assertEqual(
                tuple(helper.get_dynamic_shape().eval({
                    dim_1: 6,
                    dim_2: 10,
                    shape_1: [7, 8],
                    x: np.arange(120 * 9).reshape([-1, 4, 5, 6])
                })),
                (1, 2, 3, 4, 5, -1, 6, 7, 8, 6, 10, 11, 9, 4, 9, 4)
            )

            # test to add a shape piece with non-deterministic shape
            # which should cause error
            with self.assertRaises(ValueError) as cm:
                helper.add(tf.placeholder(dtype=tf.int32, shape=(None,)))
            self.assertIn(
                'not deterministic, which is not supported',
                str(cm.exception)
            )

            # test a realistic example to reshape tensor
            x = tf.placeholder(tf.float32, shape=[None, None, 2])
            group_size = tf.placeholder(tf.int32, shape=())
            helper = ReshapeHelper()
            helper.add(1).add([group_size, -1]).add_template(x, lambda s: s[1:])
            x_reshaped = helper.reshape(x)
            self.assertEqual(
                helper.get_static_shape().as_list(),
                [1, None, None, None, 2]
            )
            np.testing.assert_equal(
                x_reshaped.eval({
                    x: np.arange(36).reshape([6, 3, 2]),
                    group_size: 3
                }),
                np.arange(36).reshape([1, 3, 2, 3, 2])
            )

if __name__ == '__main__':
    unittest.main()
