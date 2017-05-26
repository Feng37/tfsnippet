# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.bayes import StochasticTensor, Normal, Bernoulli
from tests.bayes.distributions._helper import _MyDistribution
from tests.helper import TestCase


class StochasticTensorSkeletonTestCase(TestCase):

    def _setUp(self):
        self.distrib = _MyDistribution(
            np.arange(24, dtype=np.float32).reshape([2, 3, 4]))

    def test_non_observed_tensor(self):
        t = StochasticTensor(self.distrib)
        self.assertIs(t.distribution, self.distrib)
        self.assertEqual(t.dtype, tf.float32)
        self.assertFalse(t.is_observed)
        self.assertIsNone(t.observed_tensor)
        self.assertEqual(t, t)
        self.assertEqual(hash(t), hash(t))

    def test_observed_tensor(self):
        with self.get_session():
            distrib = _MyDistribution(
                np.arange(24, dtype=np.int32).reshape([2, 3, 4]))
            observed = np.arange(24, dtype=np.int32).reshape([2, 3, 4])
            t = StochasticTensor(distrib, observed=observed)
            self.assertIs(t.distribution, distrib)
            self.assertEqual(t.dtype, tf.int32)
            self.assertTrue(t.is_observed)
            self.assertIsInstance(t.observed_tensor, tf.Tensor)
            np.testing.assert_equal(t.observed_tensor.eval(), observed)
            self.assertEqual(t, t)
            self.assertNotEqual(StochasticTensor(distrib), t)
            self.assertEqual(hash(t), hash(t))

    def test_attributes_from_distribution(self):
        with self.get_session():
            distrib = Normal(0., 1.)
            t = StochasticTensor(distrib)

            for k in ['param_dtype', 'is_continuous', 'is_reparameterized',
                      'is_enumerable', 'enum_value_count']:
                self.assertEqual(getattr(distrib, k), getattr(t, k),
                                 msg='attribute %r mismatch.' % k)

    def test_dynamic_dimension_replaced_by_observed_shape(self):
        distrib = _MyDistribution(tf.placeholder(tf.float32, (None, 3, 4)))
        t = StochasticTensor(
            distrib,
            observed=tf.placeholder(tf.float32, (2, 3, None))
        )
        self.assertEqual(t.get_shape().as_list(), [2, 3, None])

    def test_incompatible_observed_shape(self):
        distrib = _MyDistribution(tf.placeholder(tf.float32, (None, 3, 4)))
        with self.assertRaisesRegex(
                ValueError, r'The shape of observed is .*, which is not '
                            r'compatible with the shape .* of the '
                            r'StochasticTensor\.'):
            _ = StochasticTensor(
                distrib,
                observed=np.arange(24, dtype=np.float32),
                validate_observed_shape=True
            )

    def test_construction_with_factory(self):
        t = StochasticTensor(lambda: self.distrib)
        self.assertIs(t.distribution, self.distrib)

    def test_invalid_distribution_type(self):
        with self.assertRaisesRegex(
                TypeError, 'Expected a Distribution but got .*'):
            _ = StochasticTensor('')

        with self.assertRaisesRegex(
                TypeError, 'Expected a Distribution but got .*'):
            _ = StochasticTensor(lambda: '')

    def test_shape_for_None_sample_num(self):
        with self.get_session():
            t = StochasticTensor(self.distrib)
            self.assertIsNone(t.sample_num)
            self.assertEqual(t._sample_shape, ())
            self.assertEqual(t._static_shape.as_list(), [2, 3, 4])
            self.assertEqual(t.get_shape().as_list(), [2, 3, 4])

    def test_shape_for_constant_sample_num(self):
        with self.get_session():
            t = StochasticTensor(self.distrib, sample_num=1)
            self.assertEqual(t.sample_num, 1)
            self.assertEqual(t._sample_shape, (1,))
            self.assertEqual(t.get_shape().as_list(), [1, 2, 3, 4])

            t = StochasticTensor(self.distrib, sample_num=2)
            self.assertEqual(t.sample_num, 2)
            self.assertEqual(t._sample_shape, (2,))
            self.assertEqual(t.get_shape().as_list(), [2, 2, 3, 4])

    def test_shape_for_dynamic_sample_num(self):
        with self.get_session():
            sample_num = tf.placeholder(tf.int32, ())
            t = StochasticTensor(self.distrib, sample_num=sample_num)
            self.assertIsInstance(t.sample_num, tf.Tensor)
            self.assertEqual(t.sample_num.eval({sample_num: 3}), 3)
            self.assertIsInstance(t._sample_shape, tf.Tensor)
            self.assertEqual(tuple(t._sample_shape.eval({sample_num: 3})),
                             (3,))
            self.assertEqual(t.get_shape().as_list(), [None, 2, 3, 4])

    def test_sample_num_is_invalid(self):
        with self.get_session():
            with self.assertRaisesRegex(
                    ValueError, '.*`sample_num` must be at least 1.*'):
                _ = StochasticTensor(self.distrib, sample_num=-1)

            with self.assertRaisesRegex(
                    ValueError, '.*`sample_num` must be an integer scalar.*'):
                _ = StochasticTensor(
                    self.distrib, sample_num=tf.placeholder(tf.int32, (2,)))

            with self.assertRaisesRegex(
                    ValueError, '.*`sample_num` must be an integer scalar.*'):
                _ = StochasticTensor(
                    self.distrib, sample_num=tf.placeholder(tf.float32, ()))

    def test_error_convert_to_tensor(self):
        with self.assertRaisesRegex(
                ValueError, 'Incompatible type conversion requested to '
                            'type .* for variable of type .*'):
            _ = tf.convert_to_tensor(
                StochasticTensor(Normal(0., [1., 2., 3.])),
                dtype=tf.int32
            )

    def test_construction_error(self):
        with self.assertRaisesRegex(
                ValueError, '`enum_as_samples` cannot be True when '
                            '`sample_num` is specified.'):
            _ = StochasticTensor(
                self.distrib, sample_num=1, enum_as_samples=True)

        with self.assertRaisesRegex(
                TypeError, '`enum_as_samples` is True but .* is not '
                           'enumerable.'):
            _ = StochasticTensor(Normal(0., [1., 2., 3.]), enum_as_samples=True)


class StochasticTensorRealisticTestCase(TestCase):

    def test_sampled_tensor_for_None_sample_num(self):
        with self.get_session():
            distrib = Normal(0., [1., 2., 3.])
            t = StochasticTensor(distrib)
            self.assertIsInstance(t.computed_tensor, tf.Tensor)
            self.assertEqual(t.computed_tensor.get_shape().as_list(), [3])
            self.assertIs(tf.convert_to_tensor(t), t.computed_tensor)

    def test_sampled_tensor_for_constant_sample_num(self):
        with self.get_session():
            distrib = Normal(0., [1., 2., 3.])
            t = StochasticTensor(distrib, sample_num=1)
            self.assertIsInstance(t.computed_tensor, tf.Tensor)
            self.assertEqual(t.computed_tensor.get_shape().as_list(), [1, 3])

    def test_sampled_tensor_for_dynamic_sample_num(self):
        with self.get_session():
            distrib = Normal(0., [1., 2., 3.])
            sample_num = tf.placeholder(tf.int32, ())
            t = StochasticTensor(distrib, sample_num=sample_num)
            self.assertIsInstance(t.computed_tensor, tf.Tensor)
            self.assertEqual(t.computed_tensor.get_shape().as_list(), [None, 3])

    def test_observed_tensor(self):
        with self.get_session():
            distrib = Normal(0., [1., 2., 3.])
            observed = np.asarray([[-1., 1., 2.], [0., 0., 0.]])
            t = StochasticTensor(distrib, observed=observed)
            self.assertIsInstance(t.computed_tensor, tf.Tensor)
            self.assertIs(t.computed_tensor, t.observed_tensor)
            self.assertIs(tf.convert_to_tensor(t), t.computed_tensor)
            self.assertEqual(t.computed_tensor.get_shape().as_list(), [2, 3])
            np.testing.assert_equal(t.computed_tensor.eval(), observed)

    def test_enum_as_samples(self):
        with self.get_session():
            distrib = Bernoulli(logits=[0., 1., 2.])
            t = StochasticTensor(distrib, enum_as_samples=True)
            np.testing.assert_equal(
                t.computed_tensor.eval(), [[0, 0, 0], [1, 1, 1]])

    def test_prob_for_group_event_ndims_combination(self):
        with self.get_session():
            # test distrib.group_event_ndims == None, while at the same time
            # tensor.group_event_ndims == None
            distrib = Normal(0., [1., 2., 3.])
            observed = np.asarray([[-1., 1., 2.], [0., 0., 0.]])
            t = StochasticTensor(distrib, observed=observed, sample_num=2)
            self.assertIsNone(t.group_event_ndims)
            np.testing.assert_allclose(
                t.log_prob().eval(),
                distrib.log_prob(observed, group_event_ndims=0).eval()
            )
            np.testing.assert_allclose(
                t.log_prob(group_event_ndims=0).eval(),
                distrib.log_prob(observed, group_event_ndims=0).eval()
            )
            np.testing.assert_allclose(
                t.log_prob(group_event_ndims=1).eval(),
                distrib.log_prob(observed, group_event_ndims=1).eval()
            )

            # test distrib.group_event_ndims == None, while at the same time
            # tensor.group_event_ndims == 1
            t = StochasticTensor(distrib, observed=observed, sample_num=2,
                                 group_event_ndims=1)
            self.assertEqual(t.group_event_ndims, 1)
            np.testing.assert_allclose(
                t.log_prob().eval(),
                distrib.log_prob(observed, group_event_ndims=1).eval()
            )
            np.testing.assert_allclose(
                t.log_prob(group_event_ndims=0).eval(),
                distrib.log_prob(observed, group_event_ndims=0).eval()
            )
            np.testing.assert_allclose(
                t.log_prob(group_event_ndims=1).eval(),
                distrib.log_prob(observed, group_event_ndims=1).eval()
            )

            # test distrib.group_event_ndims == 1, while at the same time
            # tensor.group_event_ndims == None
            distrib = Normal(0., [1., 2., 3.], group_event_ndims=1)
            t = StochasticTensor(distrib, observed=observed, sample_num=2)
            self.assertEqual(t.group_event_ndims, 1)
            np.testing.assert_allclose(
                t.log_prob().eval(),
                distrib.log_prob(observed, group_event_ndims=1).eval()
            )
            np.testing.assert_allclose(
                t.log_prob(group_event_ndims=0).eval(),
                distrib.log_prob(observed, group_event_ndims=0).eval()
            )
            np.testing.assert_allclose(
                t.log_prob(group_event_ndims=1).eval(),
                distrib.log_prob(observed, group_event_ndims=1).eval()
            )

    def test_prob(self):
        with self.get_session():
            distrib = Normal(0., [1., 2., 3.], group_event_ndims=1)
            observed = np.asarray([[-1., 1., 2.], [0., 0., 0.]])
            t = StochasticTensor(distrib, observed=observed, sample_num=2)
            np.testing.assert_allclose(
                t.prob().eval(),
                distrib.prob(t, group_event_ndims=1).eval()
            )
            np.testing.assert_allclose(
                t.prob(group_event_ndims=0).eval(),
                distrib.prob(t, group_event_ndims=0).eval()
            )
            np.testing.assert_allclose(
                t.prob(group_event_ndims=1).eval(),
                distrib.prob(t, group_event_ndims=1).eval()
            )

    def test_invalid_group_event_ndims(self):
        with self.assertRaisesRegex(
                TypeError, '`group_event_ndims` must be a non-negative '
                           'integer constant'):
            _ = StochasticTensor(Normal(0., 1.), group_event_ndims=-1)
        with self.assertRaisesRegex(
                TypeError, '`group_event_ndims` must be a non-negative '
                           'integer constant'):
            _ = StochasticTensor(
                Normal(0., 1.), group_event_ndims=tf.constant(0))

    def test_is_dynamic_tensor_like(self):
        with self.get_session():
            # TODO: add tests for StochasticTensor and is_dynamic_tensor_like
            #       related methods
            pass


if __name__ == '__main__':
    unittest.main()
