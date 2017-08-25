# -*- coding: utf-8 -*-
import tensorflow as tf

from .misc import (is_integer,
                   is_dynamic_tensor_like,
                   convert_to_tensor_if_dynamic)

__all__ = [
    'get_dimension_size',
    'get_dynamic_tensor_shape',
    'is_deterministic_shape',
    'explicit_broadcast',
    'maybe_explicit_broadcast',
]


def get_dimension_size(x, dim, name=None):
    """Get the dimension size of `x` at specified `dim`.

    Parameters
    ----------
    x : tf.Tensor
        The tensor whose dimension size should be queried.

    dim : int | tf.Tensor | tf.Variable
        The dimension index.

    name : str
        Optional name of this operation.

    Returns
    -------
    int | tf.Tensor
        The static or dynamic size of dimension.
    """
    with tf.name_scope(name=name, default_name='get_dimension_size',
                       values=[x, dim]):
        x = tf.convert_to_tensor(x)
        dim = convert_to_tensor_if_dynamic(dim)
        if is_dynamic_tensor_like(dim):
            return tf.shape(x)[dim]

        static_shape = x.get_shape()
        if static_shape.ndims is not None:
            value = static_shape[dim].value
            if value is not None:
                return value
        return tf.shape(x)[dim]


def get_dynamic_tensor_shape(x, slice_func=None, name=None):
    """Get the dynamic tensor shape of `x`.

    This method will return a tuple of integers (i.e., fully defined shape)
    if it is possible, despite of the method name `get_dynamic_tensor_shape`.

    Parameters
    ----------
    x : tf.Tensor
        The tensor whose shape should be queried.

    slice_func
        Optional function to slice on the dynamic shape of `x`.

        Given this argument, a tuple of integers will be returned
        whenever it is possible, even if some dimension of `x`
        discarded by `slice_func` is not deterministic.

    name : str
        Optional name of this operation.

    Returns
    -------
    tuple[int] | tf.Tensor
        Returns the shape tuple if `x` has a fully determined shape.
        Otherwise returns the shape tensor.
    """
    with tf.name_scope(name=name, default_name='get_tensor_shape',
                       values=[x]):
        static_shape = x.get_shape()
        if slice_func is not None:
            static_shape = slice_func(static_shape)
        if static_shape.is_fully_defined():
            return tuple(static_shape.as_list())

        dynamic_shape = tf.shape(x)
        if slice_func is not None:
            dynamic_shape = slice_func(dynamic_shape)
        if static_shape.ndims is not None:
            if static_shape.ndims != len(static_shape.as_list()):
                raise RuntimeError('`slice_func` returns inconsistent shapes.')
            dynamic_shape.set_shape(tf.TensorShape([static_shape.ndims]))
        return dynamic_shape


def is_deterministic_shape(x):
    """Test whether or not shape `x` is deterministic.

    A deterministic shape should be a tuple or a list of integers.
    Otherwise if `x` is an instance of `tf.TensorShape`, a `tf.Tensor`,
    or a tuple / list which contains integers but also tf.Tensor as elements,
    it should be a non-deterministic shape.

    If `x` is not any of the above listed types, a TypeError will be raised.
    """
    def delegated():
        if isinstance(x, (tuple, list)):
            for i in x:
                if i is None:
                    return False
                elif is_dynamic_tensor_like(i):
                    if not i.dtype.base_dtype.is_integer:
                        return None
                    if i.get_shape().ndims is not None and \
                            i.get_shape().ndims != 0:
                        return None
                    return False
                elif not is_integer(i):
                    return None
            return True
        elif isinstance(x, tf.TensorShape):
            return x.is_fully_defined()
        elif is_dynamic_tensor_like(x):
            if not x.dtype.base_dtype.is_integer:
                return None
            if x.get_shape().ndims is not None and x.get_shape().ndims != 1:
                return None
            return False

    ret = delegated()
    if ret is None:
        raise TypeError('%r is not a shape object.' % (x,))
    return ret


def explicit_broadcast(x, y, tail_no_broadcast_ndims=None, name=None):
    """Explicit broadcast two tensors to have the same shape.

    Parameters
    ----------
    x, y : tf.Tensor
        The tensors to broadcast.

    tail_no_broadcast_ndims : int
        If specified, this number of dimensions at the tail of the
        two tensors would not be broadcast.  This requires `x` and
        `y` to have at least this number of dimensions.

    name : str
        Optional name of this operation.

    Returns
    -------
    (tf.Tensor, tf.Tensor)
        The tensors after broadcast.
    """
    try:
        with tf.name_scope(name, default_name='explicit_broadcast'):
            def lift_shape_for_tail_no_broadcast_ndims(a, b):
                static_shape = b.get_shape()[: -tail_no_broadcast_ndims]
                static_shape = static_shape.concatenate(
                    [1] * tail_no_broadcast_ndims
                )
                if static_shape.is_fully_defined():
                    dynamic_shape = static_shape.as_list()
                else:
                    dynamic_shape = tf.concat(
                        [
                            tf.shape(b)[: -tail_no_broadcast_ndims],
                            [1] * tail_no_broadcast_ndims
                        ],
                        axis=0
                    )
                ones = tf.ones(dynamic_shape, dtype=a.dtype)
                ones.set_shape(static_shape)
                return a * ones

            def with_tail_no_broadcast_ndims():
                a = tf.convert_to_tensor(x)
                b = tf.convert_to_tensor(y)
                a = lift_shape_for_tail_no_broadcast_ndims(a, b)
                b = lift_shape_for_tail_no_broadcast_ndims(b, a)
                return a, b

            def without_tail_no_broadcast_ndims():
                a, b = x, y
                a *= tf.ones_like(b, dtype=a.dtype)
                b *= tf.ones_like(a, dtype=b.dtype)
                return a, b

            if tail_no_broadcast_ndims:
                return with_tail_no_broadcast_ndims()
            else:
                return without_tail_no_broadcast_ndims()

    except ValueError:
        raise ValueError(
            '%r and %r cannot broadcast to match. (%r vs %r)'
            % (x, y, x.get_shape(), y.get_shape())
        )


def maybe_explicit_broadcast(x, y, tail_no_broadcast_ndims=None, name=None):
    """Explicit broadcast two tensors to have the same shape if necessary.

    Parameters
    ----------
    x, y : tf.Tensor
        The tensors to broadcast.

    tail_no_broadcast_ndims : int
        If specified, this number of dimensions at the tail of the
        two tensors would not be broadcast.  This requires `x` and
        `y` to have at least this number of dimensions.

    name : str
        Optional name of this operation.

    Returns
    -------
    (tf.Tensor, tf.Tensor)
        The tensors after broadcast.
    """
    if not (x.get_shape() and y.get_shape()):
        x, y = explicit_broadcast(
            x, y, tail_no_broadcast_ndims=tail_no_broadcast_ndims,
            name=name
        )
    else:
        if x.get_shape().ndims != y.get_shape().ndims:
            x, y = explicit_broadcast(
                x, y, tail_no_broadcast_ndims=tail_no_broadcast_ndims,
                name=name
            )
        elif x.get_shape().is_fully_defined() and \
                y.get_shape().is_fully_defined():
            if x.get_shape() != y.get_shape():
                x, y = explicit_broadcast(
                    x, y, tail_no_broadcast_ndims=tail_no_broadcast_ndims,
                    name=name
                )
        else:
            x, y = explicit_broadcast(
                x, y, tail_no_broadcast_ndims=tail_no_broadcast_ndims,
                name=name
            )
    return x, y
