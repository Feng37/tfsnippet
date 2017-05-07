# -*- coding: utf-8 -*-
import unittest

import numpy as np

from tfsnippet import config
from tfsnippet.utils import floatx, set_floatx, cast_to_floatx
from tests.helper import TestCase


class ConfigUtilsTestCase(TestCase):

    def test_set_floatx(self):
        self.assertEqual(floatx(), config['FLOATX'])

        old_floatx = floatx()
        try:
            set_floatx('float16')
            self.assertEqual(floatx(), 'float16')
            self.assertEqual(config['FLOATX'], 'float16')

            set_floatx('float32')
            self.assertEqual(floatx(), 'float32')
            self.assertEqual(config['FLOATX'], 'float32')

            set_floatx('float64')
            self.assertEqual(floatx(), 'float64')
            self.assertEqual(config['FLOATX'], 'float64')

            with self.assertRaisesRegex(
                    ValueError, 'Unknown floatx type: int32'):
                set_floatx('int32')

        finally:
            set_floatx(old_floatx)

    def test_cast_to_floatx(self):
        old_floatx = floatx()
        try:
            set_floatx('float32')
            self.assertEqual(
                cast_to_floatx([1, 2, 3]).dtype,
                np.float32
            )

            set_floatx('float64')
            self.assertEqual(
                cast_to_floatx([1, 2, 3]).dtype,
                np.float64
            )

        finally:
            set_floatx(old_floatx)


if __name__ == '__main__':
    unittest.main()
