# -*- coding: utf-8 -*-
import unittest

import six
import numpy as np

from tfsnippet.utils import is_integer


class MiscTestCase(unittest.TestCase):

    def test_is_integer(self):
        if six.PY2:
            self.assertTrue(is_integer(long(1)))
        self.assertTrue(is_integer(int(1)))
        for dtype in [np.int, np.int8, np.int16, np.int32, np.int64,
                      np.uint, np.uint8, np.uint16, np.uint32, np.uint64]:
            v = np.asarray([1], dtype=dtype)[0]
            self.assertTrue(
                is_integer(v),
                msg='%r should be interpreted as integer.' % (v,)
            )
        self.assertFalse(is_integer(np.asarray(0, dtype=np.int)))
        for v in [float(1.0), '', object(), None, (), {}, []]:
            self.assertFalse(
                is_integer(v),
                msg='%r should not be interpreted as integer.' % (v,)
            )

if __name__ == '__main__':
    unittest.main()
