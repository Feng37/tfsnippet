# -*- coding: utf-8 -*-
import unittest

import six
import numpy as np
import tensorflow as tf

from tfsnippet.bayes import StochasticTensor, Normal
from tests.bayes.distributions._helper import _MyDistribution
from tests.helper import TestCase
from ._div_op import regular_div, floor_div
from ._true_div_op import true_div


class StochasticTensorTestCase(TestCase):

    def _setUp(self):
        self.distrib = _MyDistribution(
            np.arange(24, dtype=np.float32).reshape([2, 3, 4]))

    def test_equality(self):
        observed = tf.constant(0.)
        t = StochasticTensor(self.distrib, samples=observed)
        self.assertEqual(t, t)
        self.assertEqual(hash(t), hash(t))
        self.assertNotEqual(StochasticTensor(self.distrib, observed), t)

    def test_non_observed_tensor(self):
        t = StochasticTensor(self.distrib, samples=1.)
        self.assertIs(t.distribution, self.distrib)
        self.assertEqual(t.dtype, tf.float32)
        self.assertFalse(t.is_observed)
        self.assertIsInstance(t.__wrapped__, tf.Tensor)

    def test_observed_tensor(self):
        with self.get_session():
            observed = tf.constant(12345678, dtype=tf.float32)
            t = StochasticTensor(self.distrib, observed=observed)
            self.assertIs(t.distribution, self.distrib)
            self.assertEqual(t.dtype, tf.float32)
            self.assertTrue(t.is_observed)
            self.assertEqual(t.eval(), 12345678)
            self.assertIsInstance(t.__wrapped__, tf.Tensor)
            self.assertEqual(t.__wrapped__.eval(), 12345678)

    def test_attributes_from_distribution(self):
        with self.get_session():
            distrib = Normal(0., 1.)
            t = StochasticTensor(distrib, tf.constant(0.))

            for k in ['is_continuous', 'is_reparameterized', 'is_enumerable',
                      'enum_value_count']:
                self.assertEqual(getattr(distrib, k), getattr(t, k),
                                 msg='attribute %r mismatch.' % k)

    def test_dynamic_dimension_replaced_by_observed_shape(self):
        distrib = _MyDistribution(tf.placeholder(tf.float32, (None, 3, 4)))
        t = StochasticTensor(
            distrib,
            observed=tf.placeholder(tf.float32, (2, 3, None))
        )
        self.assertEqual(t.get_shape().as_list(), [2, 3, None])

    def test_dtype_replaced_by_observation(self):
        t = StochasticTensor(
            self.distrib,
            observed=tf.placeholder(tf.int32)
        )
        self.assertEqual(self.distrib.dtype, tf.float32)
        self.assertEqual(t.dtype, tf.int32)

    def test_construction_error(self):
        with self.assertRaisesRegex(
                ValueError, 'One and only one of `samples`, `observed` '
                            'should be specified.'):
            _ = StochasticTensor(self.distrib)

        with self.assertRaisesRegex(
                TypeError, '`distribution` is expected to be a Distribution '
                           'but got .*'):
            _ = StochasticTensor('', tf.constant(1.))

    def test_to_tensor(self):
        with self.get_session():
            t = StochasticTensor(self.distrib, 1.)
            self.assertIsInstance(tf.convert_to_tensor(t), tf.Tensor)
            self.assertNotIsInstance(tf.convert_to_tensor(t), StochasticTensor)

    def test_error_convert_to_tensor(self):
        with self.assertRaisesRegex(
                ValueError, 'Incompatible type conversion requested to '
                            'type .* for tensor of type .*'):
            _ = tf.convert_to_tensor(
                StochasticTensor(self.distrib, 1.),
                dtype=tf.int32
            )

    def test_session_run(self):
        with self.get_session() as sess:
            t = StochasticTensor(self.distrib, tf.constant([1, 2, 3]))
            np.testing.assert_equal(sess.run(t), [1, 2, 3])


class StochasticTensorArithTestCase(TestCase):

    def _setUp(self):
        self.distrib = _MyDistribution(
            np.arange(24, dtype=np.float32).reshape([2, 3, 4]))

    def test_prerequisite(self):
        if six.PY2:
            self.assertAlmostEqual(regular_div(3, 2), 1)
            self.assertAlmostEqual(regular_div(3.3, 1.6), 2.0625)
        else:
            self.assertAlmostEqual(regular_div(3, 2), 1.5)
            self.assertAlmostEqual(regular_div(3.3, 1.6), 2.0625)
        self.assertAlmostEqual(true_div(3, 2), 1.5)
        self.assertAlmostEqual(true_div(3.3, 1.6), 2.0625)
        self.assertAlmostEqual(floor_div(3, 2), 1)
        self.assertAlmostEqual(floor_div(3.3, 1.6), 2.0)

    def test_unary_op(self):
        def check_op(name, func, x):
            x_tensor = tf.convert_to_tensor(x)
            ans = func(x_tensor)
            res = tf.convert_to_tensor(
                func(StochasticTensor(self.distrib, x_tensor)))
            self.assertEqual(
                res.dtype, ans.dtype,
                msg='Result dtype does not match answer after unary operator '
                    '%s is applied: %r vs %r (x is %r).' %
                    (name, res.dtype, ans.dtype, x)
            )
            res_val = res.eval()
            ans_val = ans.eval()
            np.testing.assert_equal(
                res_val, ans_val,
                err_msg='Result value does not match answer after unary '
                        'operator %s is applied: %r vs %r (x is %r).' %
                        (name, res_val, ans_val, x)
            )

        with self.get_session():
            int_data = np.asarray([1, -2, 3], dtype=np.int32)
            float_data = np.asarray([1.1, -2.2, 3.3], dtype=np.float32)
            bool_data = np.asarray([True, False, True], dtype=np.bool)

            check_op('abs', abs, int_data)
            check_op('abs', abs, float_data)
            check_op('neg', (lambda v: -v), int_data)
            check_op('neg', (lambda v: -v), float_data)
            check_op('invert', (lambda v: ~v), bool_data)

    def test_binary_op(self):
        def check_op(name, func, x, y):
            x_tensor = tf.convert_to_tensor(x)
            y_tensor = tf.convert_to_tensor(y)
            ans = func(x_tensor, y_tensor)
            res_1 = tf.convert_to_tensor(
                func(StochasticTensor(self.distrib, x_tensor), y_tensor))
            res_2 = tf.convert_to_tensor(
                func(x_tensor, StochasticTensor(self.distrib, y_tensor)))
            res_3 = tf.convert_to_tensor(
                func(StochasticTensor(self.distrib, x_tensor),
                     StochasticTensor(self.distrib, y_tensor)))

            for tag, res in [('left', res_1), ('right', res_2),
                             ('both', res_3)]:
                self.assertEqual(
                    res.dtype, ans.dtype,
                    msg='Result dtype does not match answer after %s binary '
                        'operator %s is applied: %r vs %r (x is %r, y is %r).'
                        % (tag, name, res.dtype, ans.dtype, x, y)
                )
                res_val = res.eval()
                ans_val = ans.eval()
                np.testing.assert_equal(
                    res_val, ans_val,
                    err_msg='Result value does not match answer after %s '
                            'binary operator %s is applied: %r vs %r '
                            '(x is %r, y is %r).' %
                            (tag, name, res_val, ans_val, x, y)
                )

        def run_ops(x, y, ops):
            for name, func in six.iteritems(ops):
                check_op(name, func, x, y)

        arith_ops = {
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
            'div': regular_div,
            'truediv': true_div,
            'floordiv': floor_div,
            'mod': lambda x, y: x % y,
        }

        logical_ops = {
            'and': lambda x, y: x & y,
            'or': lambda x, y: x | y,
            'xor': lambda x, y: x ^ y,
        }

        relation_ops = {
            'lt': lambda x, y: x < y,
            'le': lambda x, y: x <= y,
            'gt': lambda x, y: x > y,
            'ge': lambda x, y: x >= y,
        }

        with self.get_session():
            # arithmetic operators
            run_ops(np.asarray([-4, 5, 6], dtype=np.int32),
                    np.asarray([1, -2, 3], dtype=np.int32),
                    arith_ops)
            run_ops(np.asarray([-4.4, 5.5, 6.6], dtype=np.float32),
                    np.asarray([1.1, -2.2, 3.3], dtype=np.float32),
                    arith_ops)

            # it seems that tf.pow(x, y) does not support negative integers
            # yet, so we individually test this operator here.
            check_op('pow',
                     (lambda x, y: x ** y),
                     np.asarray([-4, 5, 6], dtype=np.int32),
                     np.asarray([1, 2, 3], dtype=np.int32))
            check_op('pow',
                     (lambda x, y: x ** y),
                     np.asarray([-4.4, 5.5, 6.6], dtype=np.float32),
                     np.asarray([1.1, -2.2, 3.3], dtype=np.float32))

            # logical operators
            run_ops(np.asarray([True, False, True, False], dtype=np.bool),
                    np.asarray([True, True, False, False], dtype=np.bool),
                    logical_ops)

            # relation operators
            run_ops(np.asarray([1, -2, 3, -4, 5, 6, -4, 5, 6], dtype=np.int32),
                    np.asarray([1, -2, 3, 1, -2, 3, -4, 5, 6], dtype=np.int32),
                    relation_ops)
            run_ops(
                np.asarray([1.1, -2.2, 3.3, -4.4, 5.5, 6.6, -4.4, 5.5, 6.6],
                           dtype=np.float32),
                np.asarray([1.1, -2.2, 3.3, 1.1, -2.2, 3.3, -4.4, 5.5, 6.6],
                           dtype=np.float32),
                relation_ops
            )

    def test_getitem(self):
        def check_getitem(x, y, xx, yy):
            ans = tf.convert_to_tensor(x[y])
            res = xx[yy]

            self.assertEqual(
                res.dtype, ans.dtype,
                msg='Result dtype does not match answer after getitem '
                    'is applied: %r vs %r (x is %r, y is %r, xx is %r, '
                    'yy is %r).' % (res.dtype, ans.dtype, x, y, xx, yy)
            )
            res_val = res.eval()
            ans_val = ans.eval()
            np.testing.assert_equal(
                res_val, ans_val,
                err_msg='Result value does not match answer after '
                        'getitem is applied: %r vs %r (x is %r, y is %r, '
                        'xx is %r, yy is %r).' %
                        (res_val, ans_val, x, y, xx, yy)
            )

        class _SliceGenerator(object):
            def __getitem__(self, item):
                return item
        sg = _SliceGenerator()

        with self.get_session():
            data = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
            indices_or_slices = [
                0,
                -1,
                # TensorFlow has not supported array index yet.
                # np.asarray([0, 3, 2, 6], dtype=np.int32),
                # np.asarray([-1, -2, -3], dtype=np.int32),
                sg[0:],
                sg[:1],
                sg[:: 2],
                sg[-1:],
                sg[: -1],
                sg[:: -1],
            ]
            for s in indices_or_slices:
                x_tensor = tf.convert_to_tensor(data)
                x_simple_tensor = StochasticTensor(self.distrib, x_tensor)
                check_getitem(data, s, x_simple_tensor, s)

                if not isinstance(s, slice):
                    y_tensor = tf.convert_to_tensor(s)
                    y_simple_tensor = StochasticTensor(self.distrib, y_tensor)
                    check_getitem(data, s, x_simple_tensor, y_tensor)
                    check_getitem(data, s, x_simple_tensor, y_simple_tensor)
                    check_getitem(data, s, x_tensor, y_simple_tensor)

    def test_disallowed_op(self):
        with self.assertRaisesRegex(
                TypeError, "'Tensor' object is not iterable."):
            _ = iter(StochasticTensor(self.distrib, tf.constant(1)))

        with self.assertRaisesRegex(
                TypeError, "Using a `tf.Tensor` as a Python `bool` is not "
                           "allowed."):
            _ = not StochasticTensor(self.distrib, tf.constant(1))

        with self.assertRaisesRegex(
                TypeError, "Using a `tf.Tensor` as a Python `bool` is not "
                           "allowed."):
            if StochasticTensor(self.distrib, tf.constant(1)):
                pass


if __name__ == '__main__':
    unittest.main()
