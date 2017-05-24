# -*- coding: utf-8 -*-
import tensorflow as tf

from .misc import (is_integer,
                   is_dynamic_tensor_like,
                   convert_to_tensor_if_dynamic)
from .scope import NameScopeObject, instance_name_scope

__all__ = [
    'get_dimension_size',
    'get_dynamic_tensor_shape',
    'is_deterministic_shape',
    'ReshapeHelper',
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
                    if not i.dtype.base_dtype.is_integer or \
                            i.get_shape().ndims != 0:
                        return None
                    return False
                elif not is_integer(i):
                    return None
            return True
        elif isinstance(x, tf.TensorShape):
            return x.is_fully_defined()
        elif is_dynamic_tensor_like(x):
            if not x.dtype.base_dtype.is_integer or x.get_shape().ndims != 1:
                return None
            return False

    ret = delegated()
    if ret is None:
        raise TypeError('%r is not a shape object.' % (x,))
    return ret


class ReshapeHelper(NameScopeObject):
    """Class to help build the argument for `tf.reshape`.

    It is often the case we need to reshape a tensor with dynamic shape
    argument, while the simple `-1` trick no longer applies.
    With the help of this class, it is easy to do dynamic reshaping with
    the most proper way.  For example:

        x = tf.placeholder(tf.float32, shape=[None, None, 2])
        group_size = tf.placeholder(tf.int32, shape=())
        x_reshaped = (
            ReshapeHelper().
                add(1).
                add([group_size, -1]).
                add_template(x, lambda s: s[1:]).
                reshape(x)
        )

    Will result in a 5-d tensor `x_reshaped`, with static shape (1, ?, ?, ?, 2),
    and corresponding dynamic shape.

    Note that this reshape helper only supports to build shapes with
    determined number of dimensions.

    Parameters
    ----------
    allow_negative_one : bool
        Whether or not to allow `-1` as dimension? (default True)

    name : str
        Optional name of this reshape helper.
    """

    def __init__(self, allow_negative_one=True, name=None):
        super(ReshapeHelper, self).__init__(name=name)
        self.allow_negative_one = allow_negative_one

        # list to queue the shape pieces
        self._pieces = [[]]
        self._static_dims = []

    def __call__(self, x):
        return self.reshape(x)

    @property
    def is_deterministic(self):
        """Whether or not the `shape` is deterministic (composed of integers)?

        Note that even if `-1` is specified in the shape, it will be regarded
        as deterministic shape, since the whole shape does not depend on
        dynamic tensors.
        """
        return all(v is not None for v in self._static_dims)

    @instance_name_scope
    def get_dynamic_shape(self):
        """Get static tuple of int if deterministic, or dynamic shape tensor.

        Returns
        -------
        tuple[int] | tf.Tensor
            The shape which could be used as argument in `tf.reshape`.
        """
        if self.is_deterministic:
            assert (len(self._pieces) == 1)
            return tuple(self._pieces[0])
        else:
            return tf.concat(self._pieces, axis=0)

    def get_static_shape(self):
        """Get the static tensor shape.

        The returned tensor shape object could be used to fix the reshaped
        tensor's static shape after calling `tf.reshape`, if any dynamic
        tensors are used to build the shape.

        Note that this method will translate `-1` as undetermined dimension,
        so it is not recommended to use this if `is_deterministic` is True.

        Returns
        -------
        tf.TensorShape
            The static shape which could be used to fix the reshaped tensor.
        """
        return tf.TensorShape(
            [v if v != -1 else None for v in self._static_dims])

    @instance_name_scope
    def reshape(self, x):
        """Reshape `x` into desired shape.

        Parameters
        ----------
        x : tf.Tensor | tf.Variable
            The tensor to be reshaped.
        """
        x_reshaped = tf.reshape(x, self.get_dynamic_shape())
        if not self.is_deterministic:
            x_reshaped.set_shape(self.get_static_shape())
        return x_reshaped

    def _check_negative_one(self, *dimensions):
        neg_one_count = sum(v == -1 for v in dimensions)
        if self.allow_negative_one:
            if (-1 in self._static_dims) + neg_one_count > 1:
                raise ValueError('"-1" can only appear for at most once.')
        else:
            if neg_one_count > 0:
                raise ValueError('"-1" is not allowed.')

    def _add_with_hint(self, shape_or_dim, static_shape_hint=None):
        # pre-translate a dimension object into integer, or raise ValueError
        if isinstance(shape_or_dim, tf.Dimension):
            if shape_or_dim.value is None:
                raise ValueError(
                    '%r is an undetermined `tf.Dimension`, which is '
                    'not supported.  You should use `add_template` '
                    'instead.' % (shape_or_dim,)
                )
            shape_or_dim = shape_or_dim.value

        # main routines
        if is_integer(shape_or_dim):
            shape_or_dim = int(shape_or_dim)
            self._check_negative_one(shape_or_dim)
            self._pieces[-1].append(shape_or_dim)
            if static_shape_hint is None:
                self._static_dims.append(shape_or_dim)

        elif is_dynamic_tensor_like(shape_or_dim) and \
                shape_or_dim.get_shape().ndims == 0:
            shape_or_dim = tf.convert_to_tensor(shape_or_dim)
            self._pieces[-1].append(shape_or_dim)
            if static_shape_hint is None:
                self._static_dims.append(None)

        else:
            if is_deterministic_shape(shape_or_dim):
                if isinstance(shape_or_dim, tf.TensorShape):
                    shape_or_dim = [int(v) for v in shape_or_dim.as_list()]
                self._check_negative_one(*shape_or_dim)
                self._pieces[-1].extend(shape_or_dim)
                if static_shape_hint is None:
                    self._static_dims.extend(shape_or_dim)
            else:
                if isinstance(shape_or_dim, tf.TensorShape):
                    raise ValueError(
                        '%r is an undetermined `tf.TensorShape`, which is '
                        'not supported.  You should use `add_template` '
                        'instead.' % (shape_or_dim,)
                    )
                elif isinstance(shape_or_dim, (tuple, list)):
                    for dim in shape_or_dim:
                        if is_integer(dim):
                            dim = int(dim)
                            self._check_negative_one(dim)
                            if static_shape_hint is None:
                                self._static_dims.append(dim)
                        else:
                            if static_shape_hint is None:
                                self._static_dims.append(None)
                        self._pieces[-1].append(dim)
                else:
                    assert (is_dynamic_tensor_like(shape_or_dim))
                    assert (shape_or_dim.get_shape().ndims == 1)
                    # require at least the shape of the shape tensor is fixed
                    if not shape_or_dim.get_shape().is_fully_defined():
                        raise ValueError(
                            'The shape of component in %r is not '
                            'deterministic, which is not supported.'
                            % (shape_or_dim,)
                        )
                    shape_or_dim = tf.convert_to_tensor(shape_or_dim)
                    self._pieces.append(shape_or_dim)
                    self._pieces.append([])
                    if static_shape_hint is None:
                        for i in range(shape_or_dim.get_shape()[0].value):
                            self._static_dims.append(None)

        # use static_shape_hint to fill the static dimensions if available
        if static_shape_hint is not None:
            self._static_dims.extend(static_shape_hint)

    def add(self, shape_or_dim):
        """Add a piece of shape or a dimension.

        Parameters
        ----------
        shape_or_dim
            One of these types: int, tuple[int], list[int], tf.TensorShape,
            tf.Dimension, tf.Tensor, tf.Variable.

            If it is a constant integer or a scalar tensor, it will be
            treated as a dimension.  Otherwise it will be treated as a
            piece of shape.  In both cases, it will be appended to the
            end of the shape being built.

        Returns
        -------
        self
        """
        self._add_with_hint(shape_or_dim)
        return self

    @instance_name_scope
    def add_template(self, x, slice_func=None):
        """Add a piece of shape or a dimension, according to the shape of `x`.

        Parameters
        ----------
        x : tf.Tensor | tf.Variable
            The tensor whose shape will be used as template.

        slice_func
            Optional slicing function to slice a static or dynamic shape.

        Returns
        -------
        self
        """
        # attempt to get the deterministic shape
        static_shape = x.get_shape()
        if slice_func is not None:
            static_shape = slice_func(static_shape)

        if isinstance(static_shape, tf.TensorShape):
            static_shape = static_shape.as_list()
            if any(v is None for v in static_shape):
                dynamic_shape = tf.shape(x)
                if slice_func is not None:
                    dynamic_shape = slice_func(dynamic_shape)
                if not is_dynamic_tensor_like(dynamic_shape) \
                        or dynamic_shape.get_shape().ndims != 1:
                    raise TypeError(
                        '`slice_func` transforms shape tensor into '
                        'object %r, which is not a shape tensor.' %
                        (dynamic_shape,)
                    )
                self._add_with_hint(dynamic_shape, static_shape)
            else:
                self._add_with_hint(static_shape)

        elif isinstance(static_shape, tf.Dimension):
            assert (slice_func is not None)
            static_dim = static_shape.value
            if static_dim is None:
                dynamic_dim = slice_func(tf.shape(x))
                if not isinstance(dynamic_dim, tf.Tensor) or \
                        dynamic_dim.get_shape().ndims != 0:
                    raise TypeError(
                        '`slice_func` transforms shape tensor into '
                        'object %r, which is not a shape dimension.' %
                        (dynamic_dim,)
                    )
                self._add_with_hint(dynamic_dim)
            else:
                self._add_with_hint(static_dim)

        else:
            raise TypeError(
                '`slice_func` transforms shape into object %r, which is '
                'neither shape nor dimension.' % (static_shape,)
            )
        return self


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
                a = lift_shape_for_tail_no_broadcast_ndims(x, y)
                b = lift_shape_for_tail_no_broadcast_ndims(y, a)
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
