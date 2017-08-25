# -*- coding: utf-8 -*-
import codecs
import unittest

import os

from tests.helper import TestCase
from tfsnippet.utils import TemporaryDirectory


class TempDirTestCase(TestCase):

    def test_tempdir(self):
        def put_contents(path, content, encoding='utf-8'):
            parent_dir = os.path.split(path)[0]
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            with codecs.open(path, 'wb', encoding) as f:
                f.write(content)

        with TemporaryDirectory() as tempdir:
            self.assertEqual(os.listdir(tempdir), [])
            put_contents(os.path.join(tempdir, '1.txt'), 'abc')
            put_contents(os.path.join(tempdir, '2/3.txt'), 'def')

        self.assertFalse(os.path.exists(tempdir))

if __name__ == '__main__':
    unittest.main()
