# -*- coding: utf-8 -*-
from logging import getLogger

import six
import numpy as np
import tensorflow as tf

from tfsnippet.utils import floatx

N_SAMPLES = 10000


def big_number_verify(x, mean, stddev, n_samples, scale=5.):
    np.testing.assert_array_less(
        np.abs(x - mean), stddev * scale / np.sqrt(n_samples),
        err_msg='away from expected mean by %s stddev' % scale
    )


def get_distribution_samples(distribution_class, kwargs, n_samples=N_SAMPLES,
                             sample_shape=(), explicit_batch_size=False,
                             func=None):
    with tf.Graph().as_default():
        try:
            batch_size = n_samples if explicit_batch_size else None
            kwargs_ph = {
                k: tf.placeholder(shape=(batch_size,) + a.shape, dtype=floatx())
                for k, a in six.iteritems(kwargs)
            }
            keys = sorted(six.iterkeys(kwargs))
            distribution = distribution_class(**kwargs_ph)
            output = distribution.sample(sample_shape=sample_shape)
            prob = distribution.prob(output)
            log_prob = distribution.log_prob(output)
            outputs = [output, prob, log_prob]
            if func:
                outputs += func(distribution)
            with tf.Session() as session:
                result = session.run(outputs, feed_dict={
                    kwargs_ph[k]: np.repeat(
                        kwargs[k].reshape((-1,) + kwargs[k].shape),
                        n_samples,
                        axis=0
                    )
                    for k in keys
                })
            return tuple(np.asarray(r) for r in result)
        except Exception:
            getLogger(__name__).exception(
                'failed to get samples for %r' %
                ((distribution_class, kwargs, n_samples, sample_shape,
                  explicit_batch_size),)
            )
            raise


def compute_distribution_prob(distribution_class, kwargs, data,
                              func=None, group_event_ndims=None):
    with tf.Graph().as_default():
        try:
            kwargs_ph = {
                k: tf.placeholder(shape=a.shape, dtype=floatx())
                for k, a in six.iteritems(kwargs)
            }
            data_ph = tf.placeholder(shape=data.shape, dtype=floatx())
            keys = sorted(six.iterkeys(kwargs))
            distribution = distribution_class(**kwargs_ph)
            prob = distribution.prob(
                data, group_event_ndims=group_event_ndims)
            log_prob = distribution.log_prob(
                data, group_event_ndims=group_event_ndims)
            outputs = [prob, log_prob]
            if func:
                outputs += func(distribution)
            with tf.Session() as session:
                feed_dict = {
                    kwargs_ph[k]: kwargs[k] for k in keys
                }
                feed_dict[data_ph] = data
                result = session.run(outputs, feed_dict=feed_dict)
            return tuple(np.asarray(r) for r in result)
        except Exception:
            getLogger(__name__).exception(
                'failed to compute prob for %r' %
                ((distribution_class, kwargs),)
            )
            raise


def compute_analytic_kld(distribution_class, kwargs1, kwargs2):
    with tf.Graph().as_default():
        try:
            kwargs1_ph = {
                k: tf.placeholder(shape=a.shape, dtype=floatx())
                for k, a in six.iteritems(kwargs1)
            }
            kwargs2_ph = {
                k: tf.placeholder(shape=a.shape, dtype=floatx())
                for k, a in six.iteritems(kwargs2)
            }
            keys1 = sorted(six.iterkeys(kwargs1))
            dist1 = distribution_class(**kwargs1_ph)
            keys2 = sorted(six.iterkeys(kwargs2))
            dist2 = distribution_class(**kwargs2_ph)
            kld = dist1.analytic_kld(dist2)
            with tf.Session() as session:
                feed_dict = {}
                for k in keys1:
                    feed_dict[kwargs1_ph[k]] = kwargs1[k]
                for k in keys2:
                    feed_dict[kwargs2_ph[k]] = kwargs2[k]
                result = session.run([kld], feed_dict=feed_dict)
            return np.asarray(result[0])
        except Exception:
            getLogger(__name__).exception(
                'failed to compute prob for %r' %
                ((distribution_class, kwargs1, kwargs2),)
            )
            raise
