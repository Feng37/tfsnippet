# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.bayes import StochasticTensor, Normal
from tests.bayes.distributions._helper import _MyDistribution
from tests.helper import TestCase


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

    def test_prob_and_log_prob(self):
        with self.get_session():
            distrib = Normal(
                np.asarray(0., dtype=np.float32),
                np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
            )
            observed = np.arange(24, dtype=np.float32).reshape([4, 2, 3])
            t = StochasticTensor(distrib, observed=observed)
            np.testing.assert_almost_equal(
                t.log_prob().eval(), distrib.log_prob(observed).eval())
            np.testing.assert_almost_equal(
                t.prob().eval(), distrib.prob(observed).eval())

    def test_specialized_prob_method(self):
        class MyNormal(Normal):
            @property
            def has_specialized_prob_method(self):
                return True

        with self.get_session():
            distrib = MyNormal(
                np.asarray(0., dtype=np.float32),
                np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
            )
            observed = np.arange(24, dtype=np.float32).reshape([4, 2, 3])
            t = StochasticTensor(distrib, observed=observed)
            t.distribution._has_specialized_prob_method = True
            np.testing.assert_almost_equal(
                t.prob().eval(), distrib.prob(observed).eval())

    def test_prob_and_log_prob(self):
        with self.get_session():
            distrib = Normal(
                np.asarray(0., dtype=np.float32),
                np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                group_event_ndims=1
            )
            observed = np.arange(24, dtype=np.float32).reshape([4, 2, 3])
            t = StochasticTensor(distrib, observed=observed)
            np.testing.assert_almost_equal(
                t.prob().eval(), distrib.prob(observed).eval())
            np.testing.assert_almost_equal(
                t.log_prob().eval(), distrib.log_prob(observed).eval())

    def test_dynamic_dimension_replaced_by_observed_shape(self):
        distrib = _MyDistribution(tf.placeholder(tf.float32, (None, 3, 4)))
        t = StochasticTensor(
            distrib,
            observed=tf.placeholder(tf.float32, (2, 3, None))
        )
        self.assertEqual(t.get_shape().as_list(), [2, 3, None])

    def test_observation_dtype_cast(self):
        t = StochasticTensor(
            self.distrib,
            observed=tf.placeholder(tf.int32)
        )
        self.assertEqual(self.distrib.dtype, tf.float32)
        self.assertEqual(t.dtype, tf.float32)

    def test_disallowed_op(self):
        with self.assertRaisesRegex(
                TypeError, '`StochasticTensor` object is not iterable.'):
            _ = iter(StochasticTensor(self.distrib, tf.constant(1)))

        with self.assertRaisesRegex(
                TypeError, 'Using a `StochasticTensor` as a Python `bool` '
                           'is not allowed.'):
            _ = not StochasticTensor(self.distrib, tf.constant(1))

        with self.assertRaisesRegex(
                TypeError, 'Using a `StochasticTensor` as a Python `bool` '
                           'is not allowed.'):
            if StochasticTensor(self.distrib, tf.constant(1)):
                pass

    def test_construction_error(self):
        with self.assertRaisesRegex(
                ValueError, 'One and only one of `samples`, `observed` '
                            'should be specified.'):
            _ = StochasticTensor(self.distrib)

        with self.assertRaisesRegex(
                TypeError, '`distribution` is expected to be a Distribution '
                           'but got .*'):
            _ = StochasticTensor('', tf.constant(1.))

    def test_convert_to_tensor(self):
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

    def test_as_graph_element(self):
        g = tf.get_default_graph()
        t = StochasticTensor(self.distrib, tf.constant([1., 2., 3.]))
        self.assertIs(t._as_graph_element(), t.__wrapped__)
        self.assertIs(g.as_graph_element(t), t.__wrapped__)

        with self.assertRaisesRegex(
                RuntimeError, 'Can not convert a Tensor into a Operation.'):
            _ = t._as_graph_element(allow_tensor=False)

    def test_session_run(self):
        with self.get_session() as sess:
            t = StochasticTensor(self.distrib, tf.constant([1., 2., 3.]))
            np.testing.assert_almost_equal(sess.run(t), [1., 2., 3.])

    def test_get_attributes(self):
        t = StochasticTensor(self.distrib, tf.constant([1., 2., 3.]))
        members = dir(t)
        for member in ['dtype', 'log_prob', '__wrapped__']:
            self.assertIn(
                member, members,
                msg='%r should in dir(t), but not.' % (members,)
            )
            self.assertTrue(
                hasattr(t, member),
                msg='StochasticTensor should has member %r, but not.' %
                    (member,)
            )

    def test_set_attributes(self):
        t = StochasticTensor(self.distrib, tf.constant([1., 2., 3.]))

        self.assertTrue(hasattr(t, '_self_is_observed'))
        self.assertFalse(hasattr(t.__wrapped__, '_self_is_observed'))
        t._self_is_observed = 123
        self.assertEqual(t._self_is_observed, 123)
        self.assertFalse(hasattr(t.__wrapped__, '_self_is_observed'))

        self.assertTrue(hasattr(t, 'log_prob'))
        self.assertFalse(hasattr(t.__wrapped__, 'log_prob'))
        t.log_prob = 456
        self.assertEqual(t.log_prob, 456)
        self.assertTrue(hasattr(t, 'log_prob'))
        self.assertFalse(hasattr(t.__wrapped__, 'log_prob'))

        self.assertTrue(hasattr(t, 'get_shape'))
        self.assertTrue(hasattr(t.__wrapped__, 'get_shape'))
        t.get_shape = 789
        self.assertEqual(t.get_shape, 789)
        self.assertEqual(t.__wrapped__.get_shape, 789)
        self.assertTrue(hasattr(t, 'get_shape'))
        self.assertTrue(hasattr(t.__wrapped__, 'get_shape'))

        t.abc = 1001
        self.assertEqual(t.abc, 1001)
        self.assertEqual(t.__wrapped__.abc, 1001)
        self.assertTrue(hasattr(t, 'abc'))
        self.assertTrue(hasattr(t.__wrapped__, 'abc'))

        t.__wrapped__.xyz = 2002
        self.assertEqual(t.xyz, 2002)
        self.assertEqual(t.__wrapped__.xyz, 2002)
        self.assertTrue(hasattr(t, 'xyz'))
        self.assertTrue(hasattr(t.__wrapped__, 'xyz'))

    def test_del_attributes(self):
        t = StochasticTensor(self.distrib, tf.constant([1., 2., 3.]))

        del t._self_is_observed
        self.assertFalse(hasattr(t, '_self_is_observed'))
        self.assertFalse(hasattr(t.__wrapped__, '_self_is_observed'))

        t.abc = 1001
        del t.abc
        self.assertFalse(hasattr(t, 'abc'))
        self.assertFalse(hasattr(t.__wrapped__, 'abc'))

        t.__wrapped__.xyz = 2002
        del t.xyz
        self.assertFalse(hasattr(t, 'xyz'))
        self.assertFalse(hasattr(t.__wrapped__, 'xyz'))

        t.log_prob = 123
        del t.log_prob
        self.assertFalse(hasattr(t.__wrapped__, 'log_prob'))
        self.assertNotEqual(t.log_prob, 123)


if __name__ == '__main__':
    unittest.main()
