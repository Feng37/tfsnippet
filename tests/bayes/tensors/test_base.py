# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.bayes import StochasticTensor
from tfsnippet.distributions import Normal
from tests.distributions.test_base import _MyDistribution
from tests.helper import TestCase


class StochasticTensorTestCase(TestCase):

    def test_basic(self):
        with self.get_session():
            # test construct a non-observed tensor
            distrib = _MyDistribution(
                np.arange(24, dtype=np.float32).reshape([2, 3, 4]))
            t = StochasticTensor(distrib)
            self.assertIs(t.distribution, distrib)
            self.assertEqual(t.dtype, tf.float32)
            self.assertFalse(t.is_observed)
            self.assertIsNone(t.observed_tensor)
            self.assertEqual(t, t)
            self.assertEqual(hash(t), hash(t))

            # test construct an observed tensor
            distrib2 = _MyDistribution(
                np.arange(24, dtype=np.int32).reshape([2, 3, 4]))
            observed2 = np.arange(24, dtype=np.int32).reshape([2, 3, 4])
            t2 = StochasticTensor(distrib2, observed=observed2)
            self.assertIs(t2.distribution, distrib2)
            self.assertEqual(t2.dtype, tf.int32)
            self.assertTrue(t2.is_observed)
            self.assertIsInstance(t2.observed_tensor, tf.Tensor)
            np.testing.assert_equal(t2.observed_tensor.eval(), observed2)
            self.assertEqual(t2, t2)
            self.assertNotEqual(t, t2)
            self.assertEqual(hash(t2), hash(t2))

            # test dynamic dimension in distribution replaced by observed shape
            distrib3 = _MyDistribution(tf.placeholder(tf.float32, (None, 3, 4)))
            t3 = StochasticTensor(
                distrib3,
                observed=tf.placeholder(tf.float32, (2, 3, None))
            )
            self.assertEqual(t3.get_shape().as_list(), [2, 3, None])

            # test construct with a shape incompatible observed tensor
            with self.assertRaisesRegex(
                    ValueError, r'The shape of observed is .*, which is not '
                                r'compatible with the shape .* of the '
                                r'StochasticTensor\.'):
                _ = StochasticTensor(
                    distrib3,
                    observed=np.arange(24, dtype=np.float32),
                    validate_observed_shape=True
                )

            # test construct with a factory
            t = StochasticTensor(lambda: distrib3)
            self.assertIs(t.distribution, distrib3)

            # test construct error due to invalid distribution type
            with self.assertRaisesRegex(
                    TypeError, 'Expected a Distribution but got .*'):
                _ = StochasticTensor('')

            with self.assertRaisesRegex(
                    TypeError, 'Expected a Distribution but got .*'):
                _ = StochasticTensor(lambda: '')

    def test_sample_num_and_static_shape(self):
        with self.get_session():
            distrib = _MyDistribution(
                np.arange(24, dtype=np.float32).reshape([2, 3, 4]))

            # test sample_num == None
            t = StochasticTensor(distrib)
            self.assertIsNone(t.sample_num)
            self.assertEqual(t._sample_shape, ())
            self.assertEqual(t._static_shape.as_list(), [2, 3, 4])
            self.assertEqual(t.get_shape().as_list(), [2, 3, 4])

            # test sample_num == constant integer
            t = StochasticTensor(distrib, sample_num=1)
            self.assertEqual(t.sample_num, 1)
            self.assertEqual(t._sample_shape, (1,))
            self.assertEqual(t.get_shape().as_list(), [1, 2, 3, 4])

            t = StochasticTensor(distrib, sample_num=2)
            self.assertEqual(t.sample_num, 2)
            self.assertEqual(t._sample_shape, (2,))
            self.assertEqual(t.get_shape().as_list(), [2, 2, 3, 4])

            # test sample_num == tensor
            sample_num = tf.placeholder(tf.int32, ())
            t = StochasticTensor(distrib, sample_num=sample_num)
            self.assertIsInstance(t.sample_num, tf.Tensor)
            self.assertEqual(t.sample_num.eval({sample_num: 3}), 3)
            self.assertIsInstance(t._sample_shape, tf.Tensor)
            self.assertEqual(tuple(t._sample_shape.eval({sample_num: 3})),
                             (3,))
            self.assertEqual(t.get_shape().as_list(), [None, 2, 3, 4])

            # test invalid sample_num
            with self.assertRaisesRegex(
                    ValueError, '.*`sample_num` must be at least 1.*'):
                _ = StochasticTensor(distrib, sample_num=-1)

            with self.assertRaisesRegex(
                    ValueError, '.*`sample_num` must be an integer scalar.*'):
                _ = StochasticTensor(
                    distrib, sample_num=tf.placeholder(tf.int32, (2,)))

            with self.assertRaisesRegex(
                    ValueError, '.*`sample_num` must be an integer scalar.*'):
                _ = StochasticTensor(
                    distrib, sample_num=tf.placeholder(tf.float32, ()))

    def test_computed_tensor_and_convert_to_tensor(self):
        with self.get_session():
            distrib = Normal(0., [1., 2., 3.])
            observed = np.asarray([-1., 1., 2.])

            # test sample_num == None
            t = StochasticTensor(distrib)
            self.assertIsInstance(t.computed_tensor, tf.Tensor)
            self.assertEqual(t.computed_tensor.get_shape().as_list(), [3])
            self.assertIs(tf.convert_to_tensor(t), t.computed_tensor)

            # test sample_num == 1
            t = StochasticTensor(distrib, sample_num=1)
            self.assertIsInstance(t.computed_tensor, tf.Tensor)
            self.assertEqual(t.computed_tensor.get_shape().as_list(), [1, 3])

            # test sample_num == dynamic
            sample_num = tf.placeholder(tf.int32, ())
            t = StochasticTensor(distrib, sample_num=sample_num)
            self.assertIsInstance(t.computed_tensor, tf.Tensor)
            self.assertEqual(t.computed_tensor.get_shape().as_list(), [None, 3])

            # test observed
            t = StochasticTensor(distrib, observed=observed)
            self.assertIsInstance(t.computed_tensor, tf.Tensor)
            self.assertIs(t.computed_tensor, t.observed_tensor)
            self.assertIs(tf.convert_to_tensor(t), t.computed_tensor)
            self.assertEqual(t.computed_tensor.get_shape().as_list(), [3])
            np.testing.assert_equal(t.computed_tensor.eval(), observed)

            # test error convert_to_tensor
            with self.assertRaisesRegex(
                    ValueError, 'Incompatible type conversion requested to '
                                'type .* for variable of type .*'):
                _ = tf.convert_to_tensor(t, dtype=tf.int32)

    def test_prob_and_log_prob(self):
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
            self.assertEquals(t.group_event_ndims, 1)
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

            # test prob method
            np.testing.assert_allclose(
                t.prob().eval(),
                distrib.prob(observed, group_event_ndims=1).eval()
            )
            np.testing.assert_allclose(
                t.prob(group_event_ndims=0).eval(),
                distrib.prob(observed, group_event_ndims=0).eval()
            )
            np.testing.assert_allclose(
                t.prob(group_event_ndims=1).eval(),
                distrib.prob(observed, group_event_ndims=1).eval()
            )

            # test invalid group_event_ndims
            with self.assertRaisesRegex(
                    TypeError, '`group_event_ndims` must be a non-negative '
                               'integer constant'):
                _ = StochasticTensor(distrib, group_event_ndims=-1)
            with self.assertRaisesRegex(
                    TypeError, '`group_event_ndims` must be a non-negative '
                               'integer constant'):
                _ = StochasticTensor(distrib, group_event_ndims=tf.constant(0))

    def test_is_dynamic_tensor_like(self):
        with self.get_session():
            # TODO: add tests for StochasticTensor and is_dynamic_tensor_like
            #       related methods
            pass


if __name__ == '__main__':
    unittest.main()
