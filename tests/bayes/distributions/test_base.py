# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.bayes import StochasticTensor, Bernoulli
from tests.helper import TestCase
from tests.bayes.distributions._helper import _MyDistribution


class DistributionTestCase(TestCase):

    def _setUp(self):
        self.p_data = np.arange(1, 25, dtype=np.float32).reshape([2, 3, 4])
        self.p1 = tf.placeholder(tf.float32, (2, 3, None))
        self.p2 = tf.placeholder(tf.float32, (None, 3, 4))

        self.x_data = -(
            np.arange(1, 6, dtype=np.float32).reshape([5, 1, 1, 1]) +
            self.p_data.reshape([1, 2, 3, 4])
        )
        self.x1 = tf.placeholder(tf.float32, (5, 2, 3, None))
        self.x2 = tf.placeholder(tf.float32, (5, None, 3, 4))
        self.x3 = tf.placeholder(tf.float32, (None, 2, 3, 4))

        self.log_prob_data = np.sum(self.x_data * self.p_data, axis=-1)

    def _sample(self, sample_shape):
        p = self.p_data
        return np.tile(
            p.reshape([1] * len(sample_shape) + list(p.shape)),
            list(sample_shape) + [1] * len(p.shape)
        )

    def test_group_event_ndims_attributes(self):
        with self.get_session():
            p_data = np.arange(1, 25, dtype=np.float32).reshape([2, 3, 4])
            self.assertIsNone(_MyDistribution(p_data).group_event_ndims)
            self.assertEqual(_MyDistribution(p_data, 2).group_event_ndims, 2)

    def test_log_prob_with_static_data(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            log_prob = dist.log_prob(self.x_data)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2, 3])
            np.testing.assert_almost_equal(
                log_prob.eval(),
                self.log_prob_data
            )

            dist = _MyDistribution(self.p_data, group_event_ndims=1)
            log_prob = dist.log_prob(self.x_data)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2])
            np.testing.assert_almost_equal(
                log_prob.eval(),
                np.sum(self.log_prob_data, axis=-1)
            )

            log_prob = dist.log_prob(self.x_data, group_event_ndims=0)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2, 3])
            np.testing.assert_almost_equal(
                log_prob.eval(),
                self.log_prob_data
            )

    def test_log_prob_with_bad_shape(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            with self.assertRaisesRegex(
                    Exception, '.*Dimensions must be equal, but are 2 and 3.*'):
                dist.log_prob(self.x_data[:, :, :2, :])

            log_prob = dist.log_prob(self.x_data, group_event_ndims=2)
            self.assertEqual(log_prob.get_shape().as_list(), [5])
            np.testing.assert_almost_equal(
                log_prob.eval(),
                np.sum(np.sum(self.log_prob_data, axis=-1), axis=-1)
            )

    def test_log_prob_with_dynamic_value_shape(self):
        with self.get_session():
            dist = _MyDistribution(self.p1)
            log_prob = dist.log_prob(self.x1)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2, 3])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x1: self.x_data, self.p1: self.p_data}),
                self.log_prob_data
            )

            log_prob = dist.log_prob(self.x1, group_event_ndims=1)
            self.assertEqual(log_prob.get_shape().as_list(), [5, 2])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x1: self.x_data, self.p1: self.p_data}),
                np.sum(self.log_prob_data, axis=-1)
            )

            log_prob = dist.log_prob(self.x1, group_event_ndims=2)
            self.assertEqual(log_prob.get_shape().as_list(), [5])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x1: self.x_data, self.p1: self.p_data}),
                np.sum(np.sum(self.log_prob_data, axis=-1), axis=-1)
            )

    def test_log_prob_with_dynamic_batch_shape(self):
        with self.get_session():
            dist = _MyDistribution(self.p2)
            log_prob = dist.log_prob(self.x2)
            self.assertEqual(log_prob.get_shape().as_list(), [5, None, 3])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x2: self.x_data, self.p2: self.p_data}),
                self.log_prob_data
            )

            log_prob = dist.log_prob(self.x2, group_event_ndims=1)
            self.assertEqual(log_prob.get_shape().as_list(), [5, None])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x2: self.x_data, self.p2: self.p_data}),
                np.sum(self.log_prob_data, axis=-1)
            )

            log_prob = dist.log_prob(self.x2, group_event_ndims=2)
            self.assertEqual(log_prob.get_shape().as_list(), [5])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x2: self.x_data, self.p2: self.p_data}),
                np.sum(np.sum(self.log_prob_data, axis=-1), axis=-1)
            )

    def test_log_prob_with_dynamic_data_and_batch_size(self):
        with self.get_session():
            dist = _MyDistribution(self.p2)
            log_prob = dist.log_prob(self.x3)
            # note: x3.shape == (None, 2, 3, 4), thus log_prob.shape[1] == 2
            self.assertEqual(log_prob.get_shape().as_list(), [None, 2, 3])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x3: self.x_data, self.p2: self.p_data}),
                self.log_prob_data
            )

            log_prob = dist.log_prob(self.x3, group_event_ndims=1)
            self.assertEqual(log_prob.get_shape().as_list(), [None, 2])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x3: self.x_data, self.p2: self.p_data}),
                np.sum(self.log_prob_data, axis=-1)
            )

            log_prob = dist.log_prob(self.x3, group_event_ndims=2)
            self.assertEqual(log_prob.get_shape().as_list(), [None])
            np.testing.assert_almost_equal(
                log_prob.eval({self.x3: self.x_data, self.p2: self.p_data}),
                np.sum(np.sum(self.log_prob_data, axis=-1), axis=-1)
            )

    def test_prob_equal_to_exp_log_prob(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            prob = dist.prob(self.x_data[0, ...])
            np.testing.assert_allclose(
                prob.eval(),
                np.exp(self.log_prob_data[0, ...])
            )

    def test_sample_for_empty_shape(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            n = tf.placeholder(tf.int32)
            samples = dist.sample()
            self.assertIsNone(samples.samples_ndims)
            self.assertEqual(
                samples.get_shape().as_list(),
                list(self.p_data.shape)
            )
            np.testing.assert_almost_equal(samples.eval(), self._sample([]))

    def test_sample_for_hybird_shape(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            n = tf.placeholder(tf.int32)
            samples = dist.sample([4, n])
            self.assertEqual(samples.samples_ndims, 2)
            self.assertEqual(
                samples.get_shape().as_list(),
                [4, None] + list(self.p_data.shape)
            )
            np.testing.assert_almost_equal(
                samples.eval({n: 5}), self._sample([4, 5]))

    def test_sample_for_static_shape(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            for sample_shape in [(), (1,), (10,), (4, 5)]:
                samples = dist.sample(sample_shape)
                self.assertEqual(
                    samples.samples_ndims, len(sample_shape) or None)
                self.assertIsInstance(samples, StochasticTensor)
                self.assertEqual(
                    samples.get_shape().as_list(),
                    list(sample_shape) + list(self.p_data.shape),
                    msg='Shape for sample_shape %r is not correct.' %
                        (sample_shape,)
                )
                np.testing.assert_almost_equal(
                    samples.eval(), self._sample(sample_shape),
                    err_msg='Samples for sample_shape %r is not correct.' %
                            (sample_shape,)
                )

    def test_sample_for_dynamic_shape(self):
        with self.get_session():
            dist = _MyDistribution(self.p1)
            for sample_shape in [(), (1,), (10,), (4, 5)]:
                samples = dist.sample(tf.constant(sample_shape, dtype=tf.int32))
                self.assertEqual(
                    samples.samples_ndims, len(sample_shape) or None)
                self.assertEqual(
                    samples.get_shape().as_list(),
                    list(sample_shape) + self.p1.get_shape().as_list(),
                    msg='Shape for sample_shape %r is not correct.' %
                        (sample_shape,)
                )
                np.testing.assert_almost_equal(
                    samples.eval({self.p1: self.p_data}),
                    self._sample(sample_shape),
                    err_msg='Samples for sample_shape %r is not correct.' %
                            (sample_shape,)
                )

    def test_sample_for_fully_dynamic_shape(self):
        with self.get_session():
            p = tf.placeholder(tf.float32)
            dist = _MyDistribution(p)
            for sample_shape in [(), (1,), (10,), (4, 5)]:
                sample_shape_ph = tf.placeholder(tf.int32)
                samples = dist.sample(sample_shape_ph)
                self.assertEqual(
                    samples.samples_ndims.eval({
                        p: self.p_data, sample_shape_ph: sample_shape
                    }),
                    len(sample_shape)
                )
                self.assertIsNone(
                    samples.get_shape().ndims,
                    msg='Shape for sample_shape %r is not correct.' %
                        (sample_shape,)
                )
                np.testing.assert_almost_equal(
                    samples.eval({p: self.p_data,
                                  sample_shape_ph: sample_shape}),
                    self._sample(sample_shape),
                    err_msg='Samples for sample_shape %r is not correct.' %
                            (sample_shape,)
                )

    def test_sample_error(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            with self.assertRaisesRegex(
                    TypeError, '.* is not a shape object.'):
                dist.sample(tf.constant(1))
            with self.assertRaisesRegex(
                    TypeError, '.* is not a shape object.'):
                dist.sample('')

    def test_sample_n_None(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            samples = dist.sample_n()
            self.assertIsNone(samples.samples_ndims)
            self.assertIsInstance(samples, StochasticTensor)
            self.assertEqual(
                samples.get_shape().as_list(),
                list(self.p_data.shape)
            )
            np.testing.assert_almost_equal(
                samples.eval(), self._sample(()))

    def test_sample_n_1(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            samples = dist.sample_n(1)
            self.assertEqual(samples.samples_ndims, 1)
            self.assertEqual(
                samples.get_shape().as_list(),
                [1] + list(self.p_data.shape)
            )
            np.testing.assert_almost_equal(
                samples.eval(), self._sample([1]))

    def test_sample_n_10(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            samples = dist.sample_n(10)
            self.assertEqual(samples.samples_ndims, 1)
            self.assertEqual(
                samples.get_shape().as_list(),
                [10] + list(self.p_data.shape)
            )
            np.testing.assert_almost_equal(
                samples.eval(), self._sample([10]))

    def test_sample_n_dynamic(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            n = tf.placeholder(tf.int32, shape=())
            samples = dist.sample_n(n)
            self.assertEqual(samples.samples_ndims, 1)
            self.assertEqual(
                samples.get_shape().as_list(),
                [None] + list(self.p_data.shape)
            )
            np.testing.assert_almost_equal(
                samples.eval({n: 10}), self._sample([10]))

    def test_sample_n_fully_dynamic(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            n = tf.placeholder(tf.int32)
            samples = dist.sample_n(n)
            self.assertEqual(samples.samples_ndims, 1)
            self.assertEqual(
                samples.get_shape().as_list(),
                [None] + list(self.p_data.shape)
            )
            np.testing.assert_almost_equal(
                samples.eval({n: 10}), self._sample([10]))

    def test_sample_n_0(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            samples = dist.sample_n(0)
            self.assertEqual(samples.samples_ndims, 1)
            self.assertEqual(
                samples.get_shape().as_list(),
                [0] + list(self.p_data.shape)
            )
            np.testing.assert_almost_equal(
                samples.eval(), self._sample([0]))

    def test_sample_n_dynamic_0(self):
        with self.get_session():
            dist = _MyDistribution(self.p_data)
            n = tf.placeholder(tf.int32)
            samples = dist.sample_n(n)
            self.assertEqual(samples.samples_ndims, 1)
            self.assertEqual(
                samples.get_shape().as_list(),
                [None] + list(self.p_data.shape)
            )
            np.testing.assert_almost_equal(
                samples.eval({n: 0}), self._sample([0]))

    def test_observe(self):
        dist = _MyDistribution(self.p_data)
        samples = dist.observe(self.x_data, samples_ndims=2,
                               validate_shape=True)
        self.assertIsInstance(samples, StochasticTensor)
        self.assertEqual(samples.samples_ndims, 2)
        with self.get_session():
            np.testing.assert_almost_equal(samples.eval(), self.x_data)

    def test_sample_or_observe(self):
        dist = _MyDistribution(self.p_data)

        samples = dist.sample_or_observe(validate_shape=True)
        self.assertIsInstance(samples, StochasticTensor)
        self.assertEqual(samples.get_shape(), [2, 3, 4])
        self.assertIsNone(samples.samples_ndims)

        samples = dist.sample_or_observe(10, validate_shape=True)
        self.assertIsInstance(samples, StochasticTensor)
        self.assertEqual(samples.get_shape(), [10, 2, 3, 4])
        self.assertEqual(samples.samples_ndims, 1)

        samples = dist.sample_or_observe(10, observed=self.p_data,
                                         validate_shape=True)
        self.assertIsInstance(samples, StochasticTensor)
        self.assertEqual(samples.get_shape(), list(self.p_data.shape))
        self.assertEqual(samples.samples_ndims, 1)
        with self.get_session():
            np.testing.assert_almost_equal(samples.eval(), self.p_data)

    def test_call(self):
        dist = _MyDistribution(self.p_data)

        samples = dist(10)
        self.assertIsInstance(samples, StochasticTensor)
        self.assertEqual(samples.get_shape(), [10, 2, 3, 4])

        samples = dist(10, observed=self.p_data)
        self.assertIsInstance(samples, StochasticTensor)
        self.assertEqual(samples.get_shape(), list(self.p_data.shape))
        with self.get_session():
            np.testing.assert_almost_equal(samples.eval(), self.p_data)

    def test_override_group_event_ndims_in_sample(self):
        dist = _MyDistribution(self.p_data, group_event_ndims=1)

        self.assertEqual(
            dist.sample((1, 2)).group_event_ndims, 1)
        self.assertEqual(
            dist.sample((1, 2), group_event_ndims=0).group_event_ndims, 0)

        self.assertEqual(
            dist.sample_n(10).group_event_ndims, 1)
        self.assertEqual(
            dist.sample_n(10, group_event_ndims=2).group_event_ndims, 2)

        self.assertEqual(
            dist.observe(0.).group_event_ndims, 1)
        self.assertEqual(
            dist.observe(0., group_event_ndims=3).group_event_ndims, 3)

        self.assertEqual(
            dist.sample_or_observe().group_event_ndims, 1)
        self.assertEqual(
            dist.sample_or_observe(group_event_ndims=2).group_event_ndims, 2)

        self.assertEqual(
            dist.sample_or_observe(observed=self.p_data).group_event_ndims, 1)
        self.assertEqual(
            dist.sample_or_observe(observed=self.p_data, group_event_ndims=2).
                group_event_ndims,
            2
        )

    def test_check_numerics(self):
        distrib = _MyDistribution(self.p_data, check_numerics=True)
        x = distrib._check_numerics(tf.constant(0.) / tf.constant(0.), 'x')
        with self.get_session():
            with self.assertRaisesRegex(
                    Exception, "'x' of 'my_distribution' has nan or inf value"):
                _ = x.eval()

    def test_validate_samples_shape_static(self):
        distrib = _MyDistribution(self.p_data[:, :1, :])  # [2, 1, 4]

        # fully match
        x = tf.convert_to_tensor(self.p_data[:, :1, :])
        self.assertIs(distrib.validate_samples_shape(x), x)

        # broadcast match
        x = tf.convert_to_tensor(np.reshape([0., 1., 2.], [1, 3, 1]))
        self.assertIs(distrib.validate_samples_shape(x), x)

        # okay with auxiliary dimensions
        x = tf.convert_to_tensor(self.x_data[:, :, :1, :])
        self.assertIs(distrib.validate_samples_shape(x), x)

        # bad shape with non-broadcastable shape
        x = tf.convert_to_tensor(self.p_data[:, :, :2])
        with self.assertRaisesRegex(
                ValueError, 'The shape of `x` .* is not compatible with the '
                            'shape of distribution samples .*'):
            _ = distrib.validate_samples_shape(x)

        # bad shape with in-sufficient dimensions
        x = tf.convert_to_tensor(self.p_data[0, ...])
        with self.assertRaisesRegex(
                ValueError, 'The shape of `x` .* is not compatible with the '
                            'shape of distribution samples .*'):
            _ = distrib.validate_samples_shape(x)

    def test_validate_samples_shape_dynamic(self):
        with self.get_session():
            # static distribution params with dynamic samples
            distrib = _MyDistribution(self.p_data)
            x = tf.placeholder(tf.int32, shape=[None, 3, None])
            x_valid = distrib.validate_samples_shape(x)
            self.assertIsNot(x_valid, x)

            # fully match
            _ = x_valid.eval({x: self.p_data})
            # broadcast match
            _ = x_valid.eval({x: self.p_data[:1, :, :1]})

            with self.assertRaisesRegex(
                    Exception, 'Incompatible shapes: .*'):
                _ = x_valid.eval({x: self.p_data[:, :, :2]})

            # dynamic distribution params with static samples
            distrib = _MyDistribution(self.p1)
            x = tf.convert_to_tensor(self.p_data)
            x_valid = distrib.validate_samples_shape(x)
            self.assertIsNot(x_valid, x)

            # fully match
            _ = x_valid.eval({self.p1: self.p_data})
            # broadcast match
            _ = x_valid.eval({self.p1: self.p_data[:, :, :1]})

    def test_validate_samples_shape_fully_dynamic(self):
        with self.get_session():
            # static distribution params with dynamic samples
            p = tf.placeholder(tf.float32)
            x = tf.placeholder(tf.int32)
            distrib = _MyDistribution(p)
            x_valid = distrib.validate_samples_shape(x)
            self.assertIsNot(x_valid, x)

            # fully match
            _ = x_valid.eval({p: self.p_data, x: self.p_data})
            # broadcast match
            _ = x_valid.eval({p: self.p_data, x: self.p_data[:1, :, :1]})
            # match with auxiliary dimensions
            _ = x_valid.eval({p: self.p_data, x: self.x_data})

            with self.assertRaisesRegex(
                    Exception, 'Too few dimensions for samples to match '
                               'distribution .*'):
                _ = x_valid.eval({p: self.p_data, x: self.x_data[0, 0, ...]})

            with self.assertRaisesRegex(
                    Exception, 'Incompatible shapes: .*'):
                _ = x_valid.eval({p: self.p_data, x: self.x_data[..., :2]})


if __name__ == '__main__':
    unittest.main()
