# -*- coding: utf-8 -*-
import unittest

import numpy as np

from tfsnippet.bayes import StochasticLayer
from tests.helper import TestCase


class _MyStochasticLayer(StochasticLayer):

    def _call(self, **kwargs):
        return kwargs


class StochasticLayerTestCase(TestCase):

    def test_construction(self):
        layer = _MyStochasticLayer()
        self.assertIsNone(layer.n_samples)
        self.assertIsNone(layer.observed)
        self.assertIsNone(layer.group_event_ndims)

        layer = _MyStochasticLayer(n_samples=100, observed=np.arange(16),
                                   group_event_ndims=2)
        self.assertEqual(layer.n_samples, 100)
        np.testing.assert_equal(layer.observed, np.arange(16))
        self.assertEqual(layer.group_event_ndims, 2)

    def test_basic_inputs(self):
        output = _MyStochasticLayer()({'a': 1, 'b': 2})
        self.assertEqual(output, {'a': 1, 'b': 2, 'n_samples': None,
                                  'observed': None, 'group_event_ndims': None})

    def test_overwrite_sampling_args_by_inputs(self):
        layer = _MyStochasticLayer(n_samples=100, observed=[1, 2, 3],
                                   group_event_ndims=3)
        output = layer({
            'a': 1, 'b': 2, 'n_samples': 101, 'observed': [2, 3, 4],
            'group_event_ndims': 4
        })
        self.assertEqual(output, {
            'a': 1, 'b': 2, 'n_samples': 101, 'observed': [2, 3, 4],
            'group_event_ndims': 4
        })

    def test_overwrite_sampling_args_by_named_args(self):
        layer = _MyStochasticLayer(n_samples=100, observed=[1, 2, 3],
                                   group_event_ndims=3)
        output = layer(
            {'a': 1, 'b': 2}, n_samples=101, observed=[2, 3, 4],
            group_event_ndims=4
        )
        self.assertEqual(output, {
            'a': 1, 'b': 2, 'n_samples': 101, 'observed': [2, 3, 4],
            'group_event_ndims': 4
        })

    def test_overwrite_sampling_args_by_both(self):
        layer = _MyStochasticLayer(n_samples=100, observed=[1, 2, 3],
                                   group_event_ndims=3)
        output = layer(
            {'a': 1, 'b': 2, 'n_samples': 101, 'observed': [2, 3, 4],
             'group_event_ndims': 4},
            n_samples=102, observed=[3, 4, 5],
            group_event_ndims=5
        )
        self.assertEqual(output, {
            'a': 1, 'b': 2, 'n_samples': 101, 'observed': [2, 3, 4],
            'group_event_ndims': 4
        })

    def test_overwrite_sampling_args_with_None(self):
        layer = _MyStochasticLayer(n_samples=100, observed=[1, 2, 3],
                                   group_event_ndims=3)
        output = layer({
            'a': 1, 'b': 2, 'n_samples': None, 'observed': None,
            'group_event_ndims': None
        })
        self.assertEqual(output, {
            'a': 1, 'b': 2, 'n_samples': None, 'observed': None,
            'group_event_ndims': None
        })

    def test_error_call(self):
        layer = _MyStochasticLayer()
        with self.assertRaisesRegex(
                TypeError, '`inputs` is expected to be a dict, but got .*'):
            layer('')


if __name__ == '__main__':
    unittest.main()
