# -*- coding: utf-8 -*-
import weakref
from contextlib import contextmanager

import tensorflow as tf

from .scope import open_variable_scope

__all__ = [
    'auto_reuse_variables',
]


@contextmanager
def auto_reuse_variables(name_or_scope,
                         unique_name_scope=False,
                         **kwargs):
    """Open a variable scope, while automatically choosing `reuse` flag.
    
    The `reuse` flag will be set to False if the variable scope is opened
    for the first time, and it will be set to True each time the variable
    scope is opened again.
    
    Parameters
    ----------
    name_or_scope : str | tf.VariableScope
        The name of the variable scope, or the variable scope to open.
        
    unique_name_scope : bool
        Whether or not to open a unique name scope?

        If False, the original name scope of the variable scope will be
        reopened.  Otherwise a unique name scope will be opened.
        (default is False)

    **kwargs
        Other parameters for opening the variable scope.
        
    Yields
    ------
    tf.VariableScope
        The opened variable scope.
    """
    if 'reuse' in kwargs:
        raise RuntimeError(
            '`reuse` is not an argument of `auto_reuse_variables`.')

    # open the variable scope temporarily to capture the actual scope
    with open_variable_scope(name_or_scope, unique_name_scope=False) as vs:
        variable_scope = vs

    # check whether or not the variable scope has been initialized
    graph = tf.get_default_graph()
    if graph not in __auto_reuse_variables_graph_dict:
        __auto_reuse_variables_graph_dict[graph] = set([])
    initialized_scopes = __auto_reuse_variables_graph_dict[graph]
    reuse = variable_scope.name in initialized_scopes

    # re-enter the variable scope with proper `reuse` flag
    with open_variable_scope(name_or_scope,
                             unique_name_scope=unique_name_scope,
                             reuse=reuse,
                             **kwargs) as vs:
        yield vs
        initialized_scopes.add(vs.name)

#: dict to track the initialization state for each variable scope
#: belonging to every living graph.
__auto_reuse_variables_graph_dict = weakref.WeakKeyDictionary()
