# -*- coding: utf-8 -*-
import six
import numpy as np
import tensorflow as tf

from tfsnippet.utils import get_default_session_or_error


def big_number_verify(x, mean, stddev, scale, n_samples):
    np.testing.assert_array_less(
        np.abs(x - mean), stddev * scale / np.sqrt(n_samples),
        err_msg='away from expected mean by %s stddev' % scale
    )


class DistributionTestMixin(object):

    # inherited class should complete these attributes and methods
    dist_class = None
    simple_params = None
    extended_dimensional_params = None

    is_continuous = None
    is_reparameterized = None
    is_enumerable = None

    def get_dtype_for_param_dtype(self, param_dtype):
        return param_dtype

    def get_shapes_for_param(self, **params):
        raise NotImplementedError()

    def log_prob(self, x, group_event_ndims=None, **params):
        raise NotImplementedError()

    def prob(self, x, group_event_ndims=None, **params):
        return np.exp(self.log_prob(x, group_event_ndims=group_event_ndims,
                                    **params))

    def assert_allclose(self, a, b):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-5)

    # helper utilities
    def get_samples_and_prob(self, sample_shape=(), feed_dict=None, **params):
        tf.set_random_seed(1234)
        dist = self.dist_class(**params)
        x = dist.sample(sample_shape)
        prob, log_prob = dist.prob(x), dist.log_prob(x)
        return get_default_session_or_error().run(
            [x, prob, log_prob], feed_dict=feed_dict
        )

    # pre-configured test cases
    def test_param_dtype(self):
        dist = self.dist_class(**{
            k: np.asarray(v, dtype=np.float32)
            for k, v in six.iteritems(self.simple_params)
        })
        self.assertEqual(dist.param_dtype, tf.float32)
        self.assertEqual(dist.dtype,
                         self.get_dtype_for_param_dtype(dist.param_dtype))
        self.assertEqual(dist.sample().dtype, dist.dtype)

        dist = self.dist_class(**{
            k: np.asarray(v, dtype=np.float64)
            for k, v in six.iteritems(self.simple_params)
        })
        self.assertEqual(dist.param_dtype, tf.float64)
        self.assertEqual(dist.dtype,
                         self.get_dtype_for_param_dtype(dist.param_dtype))
        self.assertEqual(dist.sample().dtype, dist.dtype)

    def test_properties(self):
        dist = self.dist_class(**self.simple_params)
        self.assertEqual(dist.is_continuous, self.is_continuous)
        self.assertEqual(dist.is_reparameterized, self.is_reparameterized)
        self.assertEqual(dist.is_enumerable, self.is_enumerable)

    def test_shapes_with_static_parameters(self):
        with self.get_session():
            dist = self.dist_class(**self.simple_params)
            value_shape, batch_shape = \
                self.get_shapes_for_param(**self.simple_params)

            self.assertIsInstance(dist.static_value_shape, tf.TensorShape)
            self.assertEqual(dist.static_value_shape,
                             tf.TensorShape(value_shape))
            self.assertIsInstance(dist.dynamic_value_shape, tf.Tensor)
            np.testing.assert_equal(dist.dynamic_value_shape.eval(),
                                    value_shape)

            self.assertIsInstance(dist.static_batch_shape, tf.TensorShape)
            self.assertEqual(dist.static_batch_shape,
                             tf.TensorShape(batch_shape))
            self.assertIsInstance(dist.dynamic_batch_shape, tf.Tensor)
            np.testing.assert_equal(dist.dynamic_batch_shape.eval(),
                                    batch_shape)

    def test_shapes_with_dynamic_parameters(self):
        with self.get_session():
            params_ph = {
                k: tf.placeholder(tf.float32, shape=(None,) + v.shape)
                for k, v in six.iteritems(self.simple_params)
            }
            params = {
                k: np.tile(v, [3] + [1] * len(v.shape))
                for k, v in six.iteritems(self.simple_params)
            }
            feed_dict = {params_ph[k]: params[k] for k in six.iterkeys(params)}
            dist = self.dist_class(**params_ph)
            value_shape, batch_shape = self.get_shapes_for_param(**params)

            self.assertIsInstance(dist.static_value_shape, tf.TensorShape)
            self.assertEqual(dist.static_value_shape,
                             tf.TensorShape(value_shape))
            self.assertIsInstance(dist.dynamic_value_shape, tf.Tensor)
            np.testing.assert_equal(dist.dynamic_value_shape.eval(feed_dict),
                                    value_shape)

            self.assertIsInstance(dist.static_batch_shape, tf.TensorShape)
            self.assertEqual(dist.static_batch_shape[1:],
                             tf.TensorShape(batch_shape[1:]))
            self.assertIsNone(dist.static_batch_shape[0].value)
            self.assertIsInstance(dist.dynamic_batch_shape, tf.Tensor)
            np.testing.assert_equal(dist.dynamic_batch_shape.eval(feed_dict),
                                    batch_shape)

    def test_shapes_with_fully_dynamic_parameters(self):
        with self.get_session():
            params_ph = {
                k: tf.placeholder(tf.float32)
                for k, v in six.iteritems(self.simple_params)
            }
            params = {
                k: np.tile(v, [3] + [1] * len(v.shape))
                for k, v in six.iteritems(self.simple_params)
            }
            feed_dict = {params_ph[k]: params[k] for k in six.iterkeys(params)}
            dist = self.dist_class(**params_ph)
            value_shape, batch_shape = self.get_shapes_for_param(**params)

            self.assertIsInstance(dist.static_batch_shape, tf.TensorShape)
            self.assertIsNone(dist.static_batch_shape.ndims)
            self.assertIsInstance(dist.dynamic_batch_shape, tf.Tensor)
            np.testing.assert_equal(dist.dynamic_batch_shape.eval(feed_dict),
                                    batch_shape)

    def test_sampling(self):
        with self.get_session(use_gpu=True):
            x, prob, log_prob = self.get_samples_and_prob(**self.simple_params)
            value_shape, batch_shape = \
                self.get_shapes_for_param(**self.simple_params)
            np.testing.assert_equal(x.shape, batch_shape + value_shape)
            self.assert_allclose(
                prob, self.prob(x, **self.simple_params))
            self.assert_allclose(
                log_prob, self.log_prob(x, **self.simple_params))

    def test_sampling_for_static_auxiliary_batch_shape(self):
        with self.get_session(use_gpu=True):
            params = {
                k: np.tile(v, [10] + [1] * len(v.shape))
                for k, v in six.iteritems(self.simple_params)
            }
            x, prob, log_prob = self.get_samples_and_prob(**params)
            value_shape, batch_shape = \
                self.get_shapes_for_param(**self.simple_params)
            np.testing.assert_equal(
                x.shape, [10] + list(batch_shape + value_shape))
            self.assert_allclose(prob, self.prob(x, **params))
            self.assert_allclose(log_prob, self.log_prob(x, **params))

    def test_sampling_for_dynamic_shape(self):
        with self.get_session(use_gpu=True):
            params = {
                k: np.tile(v, [10] + [1] * len(v.shape))
                for k, v in six.iteritems(self.simple_params)
            }

            # sample `x` from distribution with dynamic batch shape
            tf.set_random_seed(1234)
            params_ph = {
                k: tf.placeholder(tf.float32, shape=[None] * (len(v.shape) + 1))
                for k, v in six.iteritems(self.simple_params)
            }
            feed_dict = {
                params_ph[k]: np.tile(
                    self.simple_params[k],
                    [10] + [1] * len(self.simple_params[k].shape)
                )
                for k in six.iterkeys(self.simple_params)
            }
            dist = self.dist_class(**params_ph)
            x = dist.sample()
            prob, log_prob = dist.prob(x), dist.log_prob(x)
            x, prob, log_prob = get_default_session_or_error().run(
                [x, prob, log_prob], feed_dict=feed_dict
            )

            value_shape, batch_shape = \
                self.get_shapes_for_param(**self.simple_params)
            np.testing.assert_equal(
                x.shape, [10] + list(batch_shape + value_shape))
            self.assert_allclose(prob, self.prob(x, **params))
            self.assert_allclose(log_prob, self.log_prob(x, **params))

    def test_sampling_for_fully_dynamic_shape(self):
        with self.get_session(use_gpu=True):
            params = {
                k: np.tile(v, [10] + [1] * len(v.shape))
                for k, v in six.iteritems(self.simple_params)
            }

            # sample `x` from distribution with dynamic batch shape
            tf.set_random_seed(1234)
            params_ph = {
                k: tf.placeholder(tf.float32)
                for k, v in six.iteritems(self.simple_params)
            }
            feed_dict = {
                params_ph[k]: np.tile(
                    self.simple_params[k],
                    [10] + [1] * len(self.simple_params[k].shape)
                )
                for k in six.iterkeys(self.simple_params)
            }
            dist = self.dist_class(**params_ph)
            x = dist.sample()
            prob, log_prob = dist.prob(x), dist.log_prob(x)
            x, prob, log_prob = get_default_session_or_error().run(
                [x, prob, log_prob], feed_dict=feed_dict
            )

            value_shape, batch_shape = \
                self.get_shapes_for_param(**self.simple_params)
            np.testing.assert_equal(
                x.shape, [10] + list(batch_shape + value_shape))
            self.assert_allclose(prob, self.prob(x, **params))
            self.assert_allclose(log_prob, self.log_prob(x, **params))

    def test_sampling_for_static_auxiliary_sample_shape(self):
        with self.get_session(use_gpu=True):
            x, prob, log_prob = self.get_samples_and_prob(
                sample_shape=[4, 5], **self.simple_params
            )
            value_shape, batch_shape = \
                self.get_shapes_for_param(**self.simple_params)
            np.testing.assert_equal(
                x.shape, [4, 5] + list(batch_shape + value_shape))
            self.assert_allclose(
                prob, self.prob(x, **self.simple_params))
            self.assert_allclose(
                log_prob, self.log_prob(x, **self.simple_params))

    def test_sampling_for_dynamic_auxiliary_sample_shape(self):
        with self.get_session(use_gpu=True):
            x, prob, log_prob = self.get_samples_and_prob(
                sample_shape=tf.constant([4, 5]), **self.simple_params
            )
            value_shape, batch_shape = \
                self.get_shapes_for_param(**self.simple_params)
            np.testing.assert_equal(
                x.shape, [4, 5] + list(batch_shape + value_shape))
            self.assert_allclose(
                prob, self.prob(x, **self.simple_params))
            self.assert_allclose(
                log_prob, self.log_prob(x, **self.simple_params))

    def test_sampling_for_fully_dynamic_auxiliary_sample_shape(self):
        with self.get_session(use_gpu=True):
            sample_shape_ph = tf.placeholder(tf.int32)
            x, prob, log_prob = self.get_samples_and_prob(
                sample_shape=sample_shape_ph,
                feed_dict={sample_shape_ph: [4, 5]},
                **self.simple_params
            )
            value_shape, batch_shape = \
                self.get_shapes_for_param(**self.simple_params)
            np.testing.assert_equal(
                x.shape, [4, 5] + list(batch_shape + value_shape))
            self.assert_allclose(
                prob, self.prob(x, **self.simple_params))
            self.assert_allclose(
                log_prob, self.log_prob(x, **self.simple_params))

    def test_sampling_for_1_sample_shape(self):
        with self.get_session(use_gpu=True):
            x, prob, log_prob = self.get_samples_and_prob(
                sample_shape=[1], **self.simple_params
            )
            value_shape, batch_shape = \
                self.get_shapes_for_param(**self.simple_params)
            np.testing.assert_equal(
                x.shape, [1] + list(batch_shape + value_shape))
            self.assert_allclose(
                prob, self.prob(x, **self.simple_params))
            self.assert_allclose(
                log_prob, self.log_prob(x, **self.simple_params))

    def test_sampling_for_extended_dimensional_params(self):
        with self.get_session(use_gpu=True):
            x, prob, log_prob = self.get_samples_and_prob(
                **self.extended_dimensional_params
            )
            value_shape, batch_shape = \
                self.get_shapes_for_param(**self.extended_dimensional_params)
            np.testing.assert_equal(x.shape, batch_shape + value_shape)
            self.assert_allclose(
                prob, self.prob(x, **self.extended_dimensional_params))
            self.assert_allclose(
                log_prob, self.log_prob(x, **self.extended_dimensional_params))

    def test_prob_with_group_event_ndims_0(self):
        with self.get_session(use_gpu=True):
            x, prob, log_prob = self.get_samples_and_prob(
                group_event_ndims=0, **self.extended_dimensional_params
            )
            self.assert_allclose(prob, self.prob(
                x, group_event_ndims=0, **self.extended_dimensional_params))
            self.assert_allclose(log_prob, self.log_prob(
                x, group_event_ndims=0, **self.extended_dimensional_params))

    def test_prob_with_group_event_ndims_1(self):
        with self.get_session(use_gpu=True):
            x, prob, log_prob = self.get_samples_and_prob(
                group_event_ndims=1, **self.extended_dimensional_params
            )
            self.assert_allclose(prob, self.prob(
                x, group_event_ndims=1, **self.extended_dimensional_params))
            self.assert_allclose(log_prob, self.log_prob(
                x, group_event_ndims=1, **self.extended_dimensional_params))

    def test_prob_with_higher_dimensional_params(self):
        with self.get_session(use_gpu=True):
            x, _, _ = self.get_samples_and_prob(
                **self.extended_dimensional_params
            )
            x = x[0, ...]
            dist = self.dist_class(**self.extended_dimensional_params)
            prob, log_prob = get_default_session_or_error().run(
                [dist.prob(x), dist.log_prob(x)]
            )
            self.assert_allclose(
                prob, self.prob(x, **self.extended_dimensional_params))
            self.assert_allclose(
                log_prob, self.log_prob(x, **self.extended_dimensional_params))

    def test_non_enumerable_distribution(self):
        if not self.is_enumerable:
            dist = self.dist_class(**self.simple_params)
            self.assertIsNone(dist.enum_value_count)
            msg = (
                '%s distribution is not enumerable.' %
                self.dist_class.__name__
            )
            with self.assertRaisesRegex(RuntimeError, msg):
                dist.enum_sample()


class BigNumberVerifyTestMixin(object):

    big_number_scale = 5.0
    big_number_samples = 10000

    def get_mean_stddev(self, **params):
        raise NotImplementedError()

    def test_big_number_law(self):
        with self.get_session(use_gpu=True):
            mean, stddev = self.get_mean_stddev(**self.simple_params)
            x, prob, log_prob = self.get_samples_and_prob(
                sample_shape=[self.big_number_samples],
                **self.simple_params
            )
            big_number_verify(
                np.average(x, axis=0), mean, stddev,
                self.big_number_scale, self.big_number_samples
            )

    def test_big_number_law_for_extended_dimensional_params(self):
        with self.get_session(use_gpu=True):
            mean, stddev = self.get_mean_stddev(
                **self.extended_dimensional_params
            )
            x, prob, log_prob = self.get_samples_and_prob(
                sample_shape=[self.big_number_samples],
                **self.extended_dimensional_params
            )
            for i in range(x.shape[1]):
                x_i = x[:, i, ...]
                mean_i = mean[i, ...]
                stddev_i = stddev[i, ...]
                big_number_verify(
                    np.average(x_i, axis=0), mean_i, stddev_i,
                    self.big_number_scale, self.big_number_samples
                )


class AnalyticKldTestMixin(object):

    kld_simple_params = None

    def analytic_kld(self, params1, params2):
        raise NotImplementedError()

    # pre-configured test cases
    def test_analytic_kld(self):
        with self.get_session(use_gpu=True):
            dist1 = self.dist_class(**self.simple_params)
            dist2 = self.dist_class(**self.kld_simple_params)
            kld = get_default_session_or_error().run(
                dist1.analytic_kld(dist2)
            )
            self.assert_allclose(kld, self.analytic_kld(
                self.simple_params, self.kld_simple_params))


class EnumerableTestMixin(object):

    def get_enum_value_count_for_params(self, params):
        raise NotImplementedError()

    def get_enum_samples_for_params(self, params):
        raise NotImplementedError()

    # pre-configured test cases
    def test_n_enum_values(self):
        with self.get_session():
            dist = self.dist_class(**self.simple_params)
            self.assertEqual(
                tf.convert_to_tensor(dist.enum_value_count).eval(),
                self.get_enum_value_count_for_params(self.simple_params)
            )

    def test_enum_sample(self):
        with self.get_session():
            dist = self.dist_class(**self.simple_params)
            np.testing.assert_equal(
                dist.enum_sample().eval(),
                self.get_enum_samples_for_params(self.simple_params)
            )

    def test_enum_sample_for_extended_dimensional_params(self):
        with self.get_session():
            dist = self.dist_class(**self.extended_dimensional_params)
            np.testing.assert_equal(
                dist.enum_sample().eval(),
                self.get_enum_samples_for_params(
                    self.extended_dimensional_params)
            )

    def test_enum_sample_for_dynamic_shape(self):
        with self.get_session(use_gpu=True):
            params = {
                k: np.tile(v, [10] + [1] * len(v.shape))
                for k, v in six.iteritems(self.simple_params)
            }
            params_ph = {
                k: tf.placeholder(tf.float32, shape=[None] * (len(v.shape) + 1))
                for k, v in six.iteritems(self.simple_params)
            }
            feed_dict = {
                params_ph[k]: np.tile(
                    self.simple_params[k],
                    [10] + [1] * len(self.simple_params[k].shape)
                )
                for k in six.iterkeys(self.simple_params)
            }
            dist = self.dist_class(**params_ph)
            x = dist.enum_sample().eval(feed_dict)
            np.testing.assert_equal(
                x,  self.get_enum_samples_for_params(params))

    def test_enum_sample_for_fully_dynamic_shape(self):
        with self.get_session(use_gpu=True):
            params = {
                k: np.tile(v, [10] + [1] * len(v.shape))
                for k, v in six.iteritems(self.simple_params)
            }
            params_ph = {
                k: tf.placeholder(tf.float32)
                for k, v in six.iteritems(self.simple_params)
            }
            feed_dict = {
                params_ph[k]: np.tile(
                    self.simple_params[k],
                    [10] + [1] * len(self.simple_params[k].shape)
                )
                for k in six.iterkeys(self.simple_params)
            }
            dist = self.dist_class(**params_ph)
            x = dist.enum_sample().eval(feed_dict)
            np.testing.assert_equal(
                x,  self.get_enum_samples_for_params(params))
