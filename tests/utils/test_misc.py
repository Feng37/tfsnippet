# -*- coding: utf-8 -*-
import unittest

import six
import numpy as np
import tensorflow as tf

from tfsnippet.bayes import StochasticTensor
from tfsnippet.distributions import Normal
from tfsnippet.utils import (is_integer, get_preferred_tensor_dtype, is_float,
                             is_dynamic_tensor_like,
                             convert_to_tensor_if_dynamic, MetricAccumulator)
from tests.helper import TestCase


class MiscTestCase(TestCase):

    def test_is_integer(self):
        if six.PY2:
            self.assertTrue(is_integer(long(1)))
        self.assertTrue(is_integer(int(1)))
        for dtype in [np.int, np.int8, np.int16, np.int32, np.int64,
                      np.uint, np.uint8, np.uint16, np.uint32, np.uint64]:
            v = np.asarray([1], dtype=dtype)[0]
            self.assertTrue(
                is_integer(v),
                msg='%r should be interpreted as integer.' % (v,)
            )
        self.assertFalse(is_integer(np.asarray(0, dtype=np.int)))
        for v in [float(1.0), '', object(), None, True, (), {}, []]:
            self.assertFalse(
                is_integer(v),
                msg='%r should not be interpreted as integer.' % (v,)
            )

    def test_is_float(self):
        for dtype in [float, np.float, np.float16, np.float32, np.float64,
                      np.float128]:
            v = np.asarray([1], dtype=dtype)[0]
            self.assertTrue(
                is_float(v),
                msg='%r should be interpreted as float.' % (v,)
            )
        self.assertFalse(is_integer(np.asarray(0., dtype=np.float32)))
        for v in [int(1), '', object(), None, True, (), {}, []]:
            self.assertFalse(
                is_float(v),
                msg='%r should not be interpreted as float.' % (v,)
            )

    def test_is_dynamic_tensor_like(self):
        with tf.Graph().as_default():
            for v in [tf.placeholder(tf.int32, ()),
                      tf.get_variable('v', shape=(), dtype=tf.int32),
                      StochasticTensor(Normal(0., 1.))]:
                self.assertTrue(
                    is_dynamic_tensor_like(v),
                    msg='%r should be interpreted as a dynamic tensor.' %
                        (v,)
                )
            for v in [1, 1.0, object(), (), [], {},
                      np.array([1, 2, 3])]:
                self.assertFalse(
                    is_dynamic_tensor_like(v),
                    msg='%r should not be interpreted as a dynamic tensor.' %
                        (v,)
                )

    def test_convert_to_tensor_if_dynamic(self):
        with tf.Graph().as_default():
            for v in [tf.placeholder(tf.int32, ()),
                      tf.get_variable('v', shape=(), dtype=tf.int32),
                      StochasticTensor(Normal(0., 1.))]:
                self.assertIsInstance(
                    convert_to_tensor_if_dynamic(v),
                    tf.Tensor
                )
            for v in [1, 1.0, object(), (), [], {},
                      np.array([1, 2, 3])]:
                self.assertIs(
                    convert_to_tensor_if_dynamic(v), v)

    def test_preferred_tensor_dtype(self):
        with tf.Graph().as_default():
            for dtype in [tf.int16, tf.int32, tf.float32, tf.float64]:
                ph = tf.placeholder(dtype)
                var = tf.get_variable('var_%s' % (dtype.name,), shape=(),
                                      dtype=dtype)
                self.assertEqual(
                    get_preferred_tensor_dtype(ph),
                    dtype
                )
                self.assertEqual(
                    get_preferred_tensor_dtype(var),
                    dtype
                )
            for np_dtype, tf_dtype in [(np.int16, tf.int16),
                                       (np.int32, tf.int32),
                                       (np.float32, tf.float32),
                                       (np.float64, tf.float64)]:
                array = np.asarray([], dtype=np_dtype)
                self.assertEqual(
                    get_preferred_tensor_dtype(array),
                    tf_dtype
                )
            self.assertEqual(
                get_preferred_tensor_dtype(1),
                tf.as_dtype(np.asarray([1]).dtype)
            )
            self.assertEqual(
                get_preferred_tensor_dtype(1.0),
                tf.as_dtype(np.asarray([1.0]).dtype)
            )

    def test_MetricAccumulator(self):
        # test empty initial values
        acc = MetricAccumulator()
        self.assertFalse(acc.has_value)
        self.assertEqual(acc.counter, 0)
        self.assertAlmostEqual(acc.value, 0.)
        self.assertAlmostEqual(acc.weight, 0.)

        acc.add(2)
        self.assertTrue(acc.has_value)
        self.assertEqual(acc.counter, 1)
        self.assertAlmostEqual(acc.value, 2.)
        self.assertAlmostEqual(acc.weight, 1.)

        acc.add(1, weight=3)
        self.assertTrue(acc.has_value)
        self.assertEqual(acc.counter, 2)
        self.assertAlmostEqual(acc.value, 1.25)
        self.assertAlmostEqual(acc.weight, 4.)

        acc.add(7, weight=6)
        self.assertTrue(acc.has_value)
        self.assertEqual(acc.counter, 3)
        self.assertAlmostEqual(acc.value, 4.7)
        self.assertAlmostEqual(acc.weight, 10.)

        acc.reset()
        self.assertFalse(acc.has_value)
        self.assertEqual(acc.counter, 0)
        self.assertAlmostEqual(acc.value, 0.)
        self.assertAlmostEqual(acc.weight, 0.)

        acc.add(1)
        self.assertTrue(acc.has_value)
        self.assertEqual(acc.counter, 1)
        self.assertAlmostEqual(acc.value, 1.)
        self.assertAlmostEqual(acc.weight, 1.)

        # test non-empty initial values
        acc = MetricAccumulator(1., 2.)
        self.assertFalse(acc.has_value)
        self.assertEqual(acc.counter, 0)
        self.assertAlmostEqual(acc.value, 1.)
        self.assertAlmostEqual(acc.weight, 2.)

        acc.add(1., weight=1)
        self.assertTrue(acc.has_value)
        self.assertEqual(acc.counter, 1)
        self.assertAlmostEqual(acc.value, 1.)
        self.assertAlmostEqual(acc.weight, 3.)

        acc.reset()
        self.assertFalse(acc.has_value)
        self.assertEqual(acc.counter, 0)
        self.assertAlmostEqual(acc.value, 1.)
        self.assertAlmostEqual(acc.weight, 2.)

if __name__ == '__main__':
    unittest.main()
