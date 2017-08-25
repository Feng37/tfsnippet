# -*- coding: utf-8 -*-
import unittest

import numpy as np
import six
import tensorflow as tf

from tests.helper import TestCase
from tfsnippet.bayes import StochasticTensor
from tfsnippet.bayes import Normal
from tfsnippet.utils import *


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
        float_types = [float, np.float, np.float16, np.float32, np.float64]
        for extra_type in ['float8', 'float128', 'float256']:
            if hasattr(np, extra_type):
                float_types.append(getattr(np, extra_type))
        for dtype in float_types:
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
        for v in [tf.placeholder(tf.int32, ()),
                  tf.get_variable('v', shape=(), dtype=tf.int32),
                  StochasticTensor(Normal(0., 1.), 1.)]:
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
        for v in [tf.placeholder(tf.int32, ()),
                  tf.get_variable('v', shape=(), dtype=tf.int32),
                  StochasticTensor(Normal(0., 1.), 1.)]:
            self.assertIsInstance(
                convert_to_tensor_if_dynamic(v),
                tf.Tensor
            )
        for v in [1, 1.0, object(), (), [], {},
                  np.array([1, 2, 3])]:
            self.assertIs(convert_to_tensor_if_dynamic(v), v)

    def test_preferred_tensor_dtype(self):
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
        self.assertAlmostEqual(acc.mean, 0.)
        self.assertAlmostEqual(acc.square, 0.)
        self.assertAlmostEqual(acc.var, 0.)
        self.assertAlmostEqual(acc.stddev, 0.)
        self.assertAlmostEqual(acc.weight, 0.)

        acc.add(2)
        self.assertTrue(acc.has_value)
        self.assertEqual(acc.counter, 1)
        self.assertAlmostEqual(acc.mean, 2.)
        self.assertAlmostEqual(acc.square, 4.)
        self.assertAlmostEqual(acc.var, 0.)
        self.assertAlmostEqual(acc.stddev, 0.)
        self.assertAlmostEqual(acc.weight, 1.)

        acc.add(1, weight=3)
        self.assertTrue(acc.has_value)
        self.assertEqual(acc.counter, 2)
        self.assertAlmostEqual(acc.mean, 1.25)
        self.assertAlmostEqual(acc.square, 1.75)
        self.assertAlmostEqual(acc.var, 0.1875)
        self.assertAlmostEqual(acc.stddev, 0.4330127)
        self.assertAlmostEqual(acc.weight, 4.)

        acc.add(7, weight=6)
        self.assertTrue(acc.has_value)
        self.assertEqual(acc.counter, 3)
        self.assertAlmostEqual(acc.mean, 4.7)
        self.assertAlmostEqual(acc.square, 30.1)
        self.assertAlmostEqual(acc.var, 8.01)
        self.assertAlmostEqual(acc.stddev, 2.8301943)
        self.assertAlmostEqual(acc.weight, 10.)

        acc.reset()
        self.assertFalse(acc.has_value)
        self.assertEqual(acc.counter, 0)
        self.assertAlmostEqual(acc.mean, 0.)
        self.assertAlmostEqual(acc.square, 0.)
        self.assertAlmostEqual(acc.var, 0.)
        self.assertAlmostEqual(acc.stddev, 0.)
        self.assertAlmostEqual(acc.weight, 0.)

        acc.add(1)
        self.assertTrue(acc.has_value)
        self.assertEqual(acc.counter, 1)
        self.assertAlmostEqual(acc.mean, 1.)
        self.assertAlmostEqual(acc.square, 1.)
        self.assertAlmostEqual(acc.var, 0.)
        self.assertAlmostEqual(acc.stddev, 0.)
        self.assertAlmostEqual(acc.weight, 1.)

    def test_humanize_duration(self):
        cases = [
            (0.0, '0 sec'),
            (1e-8, '1e-08 sec'),
            (0.1, '0.1 sec'),
            (1.0, '1 sec'),
            (1, '1 sec'),
            (1.1, '1.1 secs'),
            (59, '59 secs'),
            (59.9, '59.9 secs'),
            (60, '1 min'),
            (61, '1 min 1 sec'),
            (62, '1 min 2 secs'),
            (119, '1 min 59 secs'),
            (120, '2 mins'),
            (121, '2 mins 1 sec'),
            (122, '2 mins 2 secs'),
            (3599, '59 mins 59 secs'),
            (3600, '1 hr'),
            (3601, '1 hr 1 sec'),
            (3661, '1 hr 1 min 1 sec'),
            (86399, '23 hrs 59 mins 59 secs'),
            (86400, '1 day'),
            (86401, '1 day 1 sec'),
            (172799, '1 day 23 hrs 59 mins 59 secs'),
            (259199, '2 days 23 hrs 59 mins 59 secs'),
        ]
        for seconds, answer in cases:
            result = humanize_duration(seconds)
            self.assertEqual(
                result, answer,
                msg='humanize_duraion(%r) is expected to be %r, but got %r.' %
                    (seconds, answer, result)
            )

        for seconds, answer in cases[1:]:
            seconds = -seconds
            answer = answer + ' ago'
            result = humanize_duration(seconds)
            self.assertEqual(
                result, answer,
                msg='humanize_duraion(%r) is expected to be %r, but got %r.' %
                    (seconds, answer, result)
            )

    def test_unique(self):
        self.assertEqual(unique([]), [])
        self.assertEqual(unique([1, 4, 1, 3, 2, 1, 2, 3]), [1, 4, 3, 2])
        self.assertEqual(
            unique(list(range(100, 500)) + list(range(1000, 0, -1))),
            (list(range(100, 500)) + list(range(1000, 499, -1)) +
             list(range(99, 0, -1)))
        )

    def test_camel_to_underscore(self):
        def assert_convert(camel, underscore):
            self.assertEqual(
                camel_to_underscore(camel),
                underscore,
                msg='%r should be converted to %r.' % (camel, underscore)
            )

        examples = [
            ('simpleTest', 'simple_test'),
            ('easy', 'easy'),
            ('HTML', 'html'),
            ('simpleXML', 'simple_xml'),
            ('PDFLoad', 'pdf_load'),
            ('startMIDDLELast', 'start_middle_last'),
            ('AString', 'a_string'),
            ('Some4Numbers234', 'some4_numbers234'),
            ('TEST123String', 'test123_string'),
        ]
        for camel, underscore in examples:
            assert_convert(camel, underscore)
            assert_convert(underscore, underscore)
            assert_convert('_%s_' % camel, '_%s_' % underscore)
            assert_convert('_%s_' % underscore, '_%s_' % underscore)
            assert_convert('__%s__' % camel, '__%s__' % underscore)
            assert_convert('__%s__' % underscore, '__%s__' % underscore)
            assert_convert(
                '_'.join([s.capitalize() for s in underscore.split('_')]),
                underscore
            )
            assert_convert(
                '_'.join([s.upper() for s in underscore.split('_')]),
                underscore
            )


class AutoReprObjectTestCase(TestCase):

    def test_empty(self):
        self.assertEqual(repr(AutoReprObject()), 'AutoReprObject()')

        class MyObject(AutoReprObject):
            pass
        self.assertEqual(repr(MyObject()), 'MyObject()')

    def test_default_ordering(self):
        class MyObject(AutoReprObject):
            def __init__(self):
                self.b = 3
                self.a = 1
                self.aa = '2'
                self.bb = 4.0
        self.assertEqual(repr(MyObject()), "MyObject(a=1,aa='2',b=3,bb=4.0)")

    def test_manual_ordering(self):
        class MyObject(AutoReprObject):
            __repr_attributes__ = ('bb', 'aa')

            def __init__(self):
                self.b = 3
                self.a = 1
                self.aa = '2'
                self.bb = 4.0
        self.assertEqual(repr(MyObject()), "MyObject(bb=4.0,aa='2',a=1,b=3)")

    def test_class_attributes(self):
        class MyObject(AutoReprObject):
            a = 1
            b = 2
        self.assertEqual(repr(MyObject()), "MyObject()")
        MyObject.__repr_attributes__ = ('a',)
        self.assertEqual(repr(MyObject()), "MyObject(a=1)")

if __name__ == '__main__':
    unittest.main()
