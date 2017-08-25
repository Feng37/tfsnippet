# -*- coding: utf-8 -*-
import errno
import os

import six

__all__ = ['makedirs']


if six.PY2:
    def _makedirs(path, mode=0o777, exist_ok=False):
        try:
            os.makedirs(path, mode=mode)
        except OSError as exc:
            if not exist_ok or \
                    not (exc.errno == errno.EEXIST and os.path.isdir(path)):
                raise

    def _stat(path, follow_symlinks=None):
        if follow_symlinks:
            return os.stat(os.path.realpath(path))
        else:
            return os.stat(path)

    makedirs = _makedirs
else:
    makedirs = os.makedirs
