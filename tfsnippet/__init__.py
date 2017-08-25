# -*- coding: utf-8 -*-

__version__ = '0.1'


def _init_config():
    import sys
    from . import defconfig

    config = {}
    for k in dir(defconfig):
        if k.isupper() and not k.startswith('_'):
            config[k] = getattr(defconfig, k)
    del sys.modules[__name__]._init_config
    return config

#: The global configuration object.
config = _init_config()
