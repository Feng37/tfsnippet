# -*- coding: utf-8 -*-
import sys
import unittest
from contextlib import contextmanager

import tensorflow as tf

__all__ = ['TestCase']


class TestCase(unittest.TestCase):
    """Extended TestCase class."""

    @contextmanager
    def get_session(self, use_gpu=False):
        config = tf.ConfigProto(
            device_count={'GPU': int(use_gpu)}
        )
        session = tf.Session(config=config)
        with session.as_default():
            yield session

    def setUp(self):
        self.graph = tf.Graph()
        try:
            self.graph_ctx = self.graph.as_default()
            self.graph_ctx.__enter__()
        except Exception:
            self.graph_ctx = None
        else:
            self._setUp()

    def tearDown(self):
        try:
            self._tearDown()
        finally:
            try:
                if self.graph_ctx:
                    self.graph_ctx.__exit__(*sys.exc_info())
            finally:
                self.graph_ctx = None
                self.graph = None

    def _setUp(self):
        pass

    def _tearDown(self):
        pass

if not hasattr(TestCase, 'assertRaisesRegex'):
    TestCase.assertRaisesRegex = TestCase.assertRaisesRegexp
