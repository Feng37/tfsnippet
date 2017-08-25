# -*- coding: utf-8 -*-
import codecs
import os
import unittest

from tests.helper import TestCase
from tfsnippet.utils import *


class OsUtilsTestCase(TestCase):

    def test_make_dirs(self):
        with TemporaryDirectory() as tempdir:
            p1 = os.path.join(tempdir, '1/2')

            self.assertFalse(os.path.exists(p1))
            makedirs(p1)
            self.assertTrue(os.path.exists(p1))

            with self.assertRaisesRegexp(OSError, '.*exists.*'):
                makedirs(p1)
            makedirs(p1, exist_ok=True)

            p2 = os.path.join(tempdir, '2.txt')
            with codecs.open(p2, 'wb', 'utf-8') as f:
                f.write('123')
            with self.assertRaisesRegexp(OSError, '.*exists.*'):
                makedirs(p2, exist_ok=True)

if __name__ == '__main__':
    unittest.main()
