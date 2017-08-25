# -*- coding: utf-8 -*-
import re

import six
import tensorflow as tf

from tfsnippet.utils import lagacy_default_name_arg
from .base import Component

__all__ = ['DictMapper']

_VALID_KEY_FOR_DICT_MAPPER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


class DictMapper(Component):
    """Component that maps inputs to a dict of outputs.

    This component is a branch component, which maps the inputs to a dict
    of outputs, according to the specified `mapper`.  The keys of the
    mapper should be valid Python identifiers, that is, matching the
    pattern "^[A-Za-z_][A-Za-z0-9_]*$".  The values of the mapper should
    be other functions or components which produces corresponding output
    given the inputs.

    A typical usage of this `DictMapper` is to derive the distribution
    parameters for a `StochasticLayer`, for example:

        from tfsnippet.components import Sequential, Dense, Linear, DictMapper
        from tfsnippet.bayes import NormalLayer

        net = Sequential([
            Dense(100),
            DictMapper({
                'mean': Linear(2),
                'logstd': Linear(2),
            }),
            NormalLayer()
        ])

    In the above example, `net` will produce a 2-dimensional `StochasticTensor`
    of `Normal` distribution, with the mean and logstd of the distribution
    derived from two fully-connected layers, sharing the 100-unit hidden layer.

    Parameters
    ----------
    mapper : dict[str, (*args, **kwargs) -> any]
        A dict of mappers, which produces corresponding output given
        the layer inputs.

    name, scope : str
        Optional name and scope of this `DictMapper` component.
    """

    @lagacy_default_name_arg
    def __init__(self, mapper, name=None, scope=None):
        for k in six.iterkeys(mapper):
            if not _VALID_KEY_FOR_DICT_MAPPER.match(k):
                raise ValueError('The key for `DictMapper` must be a valid '
                                 'Python identifier (matching the pattern '
                                 '"^[A-Za-z_][A-Za-z0-9_]*$").')

        super(DictMapper, self).__init__(name=name, scope=scope)
        self._mapper = mapper

    def _call(self, *args, **kwargs):
        ret = {}
        for k, v in six.iteritems(self._mapper):
            with tf.variable_scope(k):
                ret[k] = v(*args, **kwargs)
        return ret
