# -*- coding: utf-8 -*-

__version__ = '0.1'


def _init_config():
    import os
    import sys
    from flask import Config
    from . import defconfig

    config = Config(os.path.abspath(os.path.curdir))
    config.from_object(defconfig)
    del sys.modules[__name__]._init_config
    return config

#: The global configuration object.
config = _init_config()
