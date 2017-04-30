# -*- coding: utf-8 -*-
import tensorflow as tf

from .misc import is_integer
from .scope import NameScopeObject, instance_name_scope

__all__ = [
    'get_dimension_size',
    'get_dynamic_tensor_shape',
    'is_deterministic_shape',
    'ReshapeHelper',
    'repeat_tensor_for_samples'
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
        if isinstance(dim, (tf.Tensor, tf.Variable)):
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
                if i is None or isinstance(i, (tf.Tensor, tf.Variable)):
                    if not i.dtype.base_dtype.is_integer or \
                                    i.get_shape().ndims != 0:
                        return None
                    return False
                elif not is_integer(i):
                    return None
            return True
        elif isinstance(x, tf.TensorShape):
            return x.is_fully_defined()
        elif isinstance(x, (tf.Tensor, tf.Variable)):
            if not x.dtype.base_dtype.is_integer or x.get_shape().ndims != 1:
                return None
            return False

    ret = delegated()
    if ret is None:
        raise TypeError('%r is not a shape object.' % (x,))
    return ret


def repeat_tensor_for_samples(x, sample_size, batch_size, name=None):
    """Repeat the tensor `x` whose first-dimension should be 1 or 
    `sample_size` or `batch_size` to a repeated tensor whose first-dimension 
    will be `sample_size` * `batch_size`.
    
    Parameters
    ----------
    x: tf.Tensor | tf.Variable
        The tensor or variable need to be repeated. 
        
    sample_size: int | tf.Tensor | tf.Variable
        The size of samples, which should be an integer scalar.
        
    batch_size: int | tf.Tensor | tf.Variable
        The size of batch, which should be an integer scalar.
        
    name : str
        Optional name of this operation.

    Returns
    -------
    tf.Tensor
        The repeated tensor generated from `x`.
    """
    def check_size_arg(n, v):
        flag = True
        if isinstance(v, (tf.Tensor, tf.Variable)):
            if v.get_shape().ndims != 0 or not v.dtype.is_integer:
                flag = False
        elif not is_integer(v):
            flag = False
        if not flag:
            raise ValueError('`%s` must be a scalar integer.' % n)

    check_size_arg('sample_size', sample_size)
    check_size_arg('batch_size', batch_size)

    with tf.name_scope(name=name, default_name='repeat_tensor_for_samples',
                       values=[x, sample_size, batch_size]):
        if not isinstance(x, (tf.Variable, tf.Tensor)):
            x = tf.convert_to_tensor(x)
        if x.get_shape().ndims == 0:
            raise ValueError('`x` must be a tensor, not a scalar.')
        if x.get_shape().ndims is None:
            raise ValueError('The number of dimensions of `x` is not '
                             'deterministic, which is not supported.')

        # Check the validation of the shape of `x` statically
        first_dim = get_dimension_size(x, 0)
        batch_size_mismatch_msg = (
            'first dimension of `x` does not match `batch_size`'
        )
        if isinstance(first_dim, int) and isinstance(batch_size, int):
            if first_dim != 1 and first_dim != batch_size:
                raise ValueError(batch_size_mismatch_msg)

        # fast routine: first_dim === 1, just repeat `x` at first dim
        if first_dim == 1:
            multiples = (
                [sample_size * batch_size] + [1] * (x.get_shape().ndims - 1)
            )
            if not is_deterministic_shape(multiples):
                multiples = tf.stack(multiples)
            return tf.tile(x, multiples)

        # slow routine: first_dim is dynamic or != 1, tile the two dimensions
        else:
            assert_ops = []
            if isinstance(first_dim, (tf.Tensor, tf.Variable)) or \
                    isinstance(batch_size, (tf.Tensor, tf.Variable)):
                assert_ops = [
                    tf.Assert(
                        tf.logical_or(
                            tf.equal(first_dim, 1),
                            tf.equal(first_dim, batch_size)
                        ),
                        [batch_size_mismatch_msg, first_dim, batch_size]
                    )
                ]
                batch_repeat = tf.cond(
                    tf.equal(first_dim, 1),
                    lambda: tf.convert_to_tensor(batch_size),
                    lambda: tf.convert_to_tensor(1)
                )
            else:
                # first dimension != 1, and it's ensured to match `batch_size`.
                batch_repeat = 1

            def f(ret):
                ret = tf.expand_dims(ret, axis=0)
                multiple_shape = (
                    [sample_size, batch_repeat] +
                    [1] * (ret.get_shape().ndims - 2)
                )
                if not is_deterministic_shape(multiple_shape):
                    multiple_shape = tf.stack(multiple_shape)

                ret = tf.tile(ret, multiple_shape)

                helper = ReshapeHelper()
                helper.add(sample_size * batch_size)
                helper.add_template(ret, lambda s: s[2:])
                return helper.reshape(ret)

            if assert_ops:
                with tf.control_dependencies(assert_ops):
                    return f(x)
            return f(x)


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
            self._static_dims.append(shape_or_dim)

        elif isinstance(shape_or_dim, (tf.Variable, tf.Tensor)) and \
                shape_or_dim.get_shape().ndims == 0:
            self._pieces[-1].append(shape_or_dim)
            self._static_dims.append(None)

        else:
            if is_deterministic_shape(shape_or_dim):
                if isinstance(shape_or_dim, tf.TensorShape):
                    shape_or_dim = [int(v) for v in shape_or_dim.as_list()]
                self._check_negative_one(*shape_or_dim)
                self._pieces[-1].extend(shape_or_dim)
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
                            self._static_dims.append(dim)
                        else:
                            self._static_dims.append(None)
                        self._pieces[-1].append(dim)
                else:
                    assert (isinstance(shape_or_dim, (tf.Variable, tf.Tensor)))
                    assert (shape_or_dim.get_shape().ndims == 1)
                    # require at least the shape of the shape tensor is fixed
                    if not shape_or_dim.get_shape().is_fully_defined():
                        raise ValueError(
                            'The shape of component in %r is not '
                            'deterministic, which is not supported.'
                            % (shape_or_dim,)
                        )
                    self._pieces.append(shape_or_dim)
                    self._pieces.append([])
                    for i in range(shape_or_dim.get_shape()[0].value):
                        self._static_dims.append(None)

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
        shape = x.get_shape()
        if slice_func is not None:
            shape = slice_func(shape)

        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()
            if any(v is None for v in shape):
                dynamic_shape = tf.shape(x)
                if slice_func is not None:
                    dynamic_shape = slice_func(dynamic_shape)
                if not isinstance(dynamic_shape, tf.Tensor) \
                        or dynamic_shape.get_shape().ndims != 1:
                    raise TypeError(
                        '`slice_func` transforms shape tensor into '
                        'object %r, which is not a shape tensor.' %
                        (dynamic_shape,)
                    )
                for i, v in enumerate(shape):
                    if v is None:
                        shape[i] = dynamic_shape[i]
            return self.add(shape)

        elif isinstance(shape, tf.Dimension):
            assert (slice_func is not None)
            shape = shape.value
            if shape is None:
                dynamic_shape = slice_func(tf.shape(x))
                if not isinstance(dynamic_shape, tf.Tensor) or \
                        dynamic_shape.get_shape().ndims != 0:
                    raise TypeError(
                        '`slice_func` transforms shape tensor into '
                        'object %r, which is not a shape dimension.' %
                        (dynamic_shape,)
                    )
                shape = dynamic_shape
            return self.add(shape)

        else:
            raise TypeError(
                '`slice_func` transforms shape into object %r, which is '
                'neither shape nor dimension.' % (shape,)
            )
