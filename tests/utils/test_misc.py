# -*- coding: utf-8 -*-
import unittest

import six
import numpy as np
import tensorflow as tf

from tfsnippet.utils import is_integer, get_preferred_tensor_dtype


class MiscTestCase(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()
