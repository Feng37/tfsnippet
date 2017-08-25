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

            for k in ['is_continuous', 'is_reparameterized']:
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

    def test_log_prob_with_group_events_ndims(self):
        with self.get_session():
            distrib = Normal(
                np.asarray(0., dtype=np.float32),
                np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                group_event_ndims=1
            )
            observed = np.asarray([[-1., 1., 2.], [0., 0., 0.]])
            t = StochasticTensor(distrib, observed=observed)
            np.testing.assert_allclose(
                t.log_prob(group_event_ndims=0).eval(),
                distrib.log_prob(t, group_event_ndims=0).eval()
            )
            np.testing.assert_allclose(
                t.log_prob(group_event_ndims=1).eval(),
                distrib.log_prob(t, group_event_ndims=1).eval()
            )

    def test_prob_with_group_events_ndims(self):
        with self.get_session():
            distrib = Normal(
                np.asarray(0., dtype=np.float32),
                np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                group_event_ndims=1
            )
            observed = np.asarray([[-1., 1., 2.], [0., 0., 0.]])
            t = StochasticTensor(distrib, observed=observed)
            np.testing.assert_allclose(
                t.prob(group_event_ndims=0).eval(),
                distrib.prob(t, group_event_ndims=0).eval()
            )
            np.testing.assert_allclose(
                t.prob(group_event_ndims=1).eval(),
                distrib.prob(t, group_event_ndims=1).eval()
            )

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

    def test_construction_error(self):
        with self.assertRaisesRegex(
                ValueError, 'One and only one of `samples`, `observed` '
                            'should be specified.'):
            _ = StochasticTensor(self.distrib)

        with self.assertRaisesRegex(
                TypeError, '`distribution` is expected to be a Distribution '
                           'but got .*'):
            _ = StochasticTensor('', tf.constant(1.))


if __name__ == '__main__':
    unittest.main()
