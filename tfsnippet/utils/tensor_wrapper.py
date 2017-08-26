# -*- coding: utf-8 -*-
import six
import tensorflow as tf
from tensorflow.python.client.session import \
    register_session_run_conversion_functions

__all__ = ['TensorWrapper', 'register_tensor_wrapper_class']


class TensorWrapper(object):
    """Tensor-like object that wraps a `tf.Tensor` instance.

    The derived classes must call `register_rensor_wrapper` so as to
    register that class into TensorFlow type system.

    Parameters
    ----------
    wrapped : tf.Tensor | TensorWrapper
        The tensor to be wrapped.
    """

    def __init__(self, wrapped):
        if isinstance(wrapped, TensorWrapper):
            wrapped = wrapped.__wrapped__
        if not isinstance(wrapped, tf.Tensor):
            raise TypeError('%r is not an instance of `tf.Tensor`.' %
                            (wrapped,))
        self.__wrapped__ = wrapped

    # mimic `tf.Tensor` interface
    def __dir__(self):
        if six.PY3:
            ret = object.__dir__(self)
        else:
            # code is based on
            # http://www.quora.com/How-dir-is-implemented-Is-there-any-PEP-related-to-that
            def get_attrs(obj):
                import types
                if not hasattr(obj, '__dict__'):
                    return []  # slots only
                if not isinstance(obj.__dict__, (dict, types.DictProxyType)):
                    raise TypeError("%s.__dict__ is not a dictionary"
                                    "" % obj.__name__)
                return obj.__dict__.keys()

            def dir2(obj):
                attrs = set()
                if not hasattr(obj, '__bases__'):
                    # obj is an instance
                    if not hasattr(obj, '__class__'):
                        # slots
                        return sorted(get_attrs(obj))
                    klass = obj.__class__
                    attrs.update(get_attrs(klass))
                else:
                    # obj is a class
                    klass = obj

                for cls in klass.__bases__:
                    attrs.update(get_attrs(cls))
                    attrs.update(dir2(cls))
                attrs.update(get_attrs(obj))
                return list(attrs)

            ret = dir2(self)

        ret = list(set(dir(self.__wrapped__) + ret))
        return ret

    def __getattr__(self, name):
        return getattr(self.__wrapped__, name)

    def __setattr__(self, name, value):
        if name.startswith('_self_') or name == '__wrapped__':
            object.__setattr__(self, name, value)
        elif hasattr(type(self), name):
            object.__setattr__(self, name, value)
        else:
            setattr(self.__wrapped__, name, value)

    def __delattr__(self, name):
        if name.startswith('_self_'):
            object.__delattr__(self, name)
        elif hasattr(type(self), name):
            object.__delattr__(self, name)
        else:
            delattr(self.__wrapped__, name)

    def __iter__(self):
        raise TypeError('`%s` object is not iterable.' %
                        self.__class__.__name__)

    def __bool__(self):
        raise TypeError(
            'Using a `%s` as a Python `bool` is not allowed. '
            'Use `if t is not None:` instead of `if t:` to test if a '
            'tensor is defined, and use TensorFlow ops such as '
            'tf.cond to execute subgraphs conditioned on the value of '
            'a tensor.' % self.__class__.__name__
        )

    def __nonzero__(self):
        raise TypeError(
            'Using a `%s` as a Python `bool` is not allowed. '
            'Use `if t is not None:` instead of `if t:` to test if a '
            'tensor is defined, and use TensorFlow ops such as '
            'tf.cond to execute subgraphs conditioned on the value of '
            'a tensor.' % self.__class__.__name__
        )

    # overloading arithmetic operations
    def __abs__(self):
        return tf.abs(self)

    def __neg__(self):
        return tf.negative(self)

    def __add__(self, other):
        return tf.add(self, other)

    def __radd__(self, other):
        return tf.add(other, self)

    def __sub__(self, other):
        return tf.subtract(self, other)

    def __rsub__(self, other):
        return tf.subtract(other, self)

    def __mul__(self, other):
        return tf.multiply(self, other)

    def __rmul__(self, other):
        return tf.multiply(other, self)

    def __div__(self, other):
        return tf.div(self, other)

    def __rdiv__(self, other):
        return tf.div(other, self)

    def __truediv__(self, other):
        return tf.truediv(self, other)

    def __rtruediv__(self, other):
        return tf.truediv(other, self)

    def __floordiv__(self, other):
        return tf.floordiv(self, other)

    def __rfloordiv__(self, other):
        return tf.floordiv(other, self)

    def __mod__(self, other):
        return tf.mod(self, other)

    def __rmod__(self, other):
        return tf.mod(other, self)

    def __pow__(self, other):
        return tf.pow(self, other)

    def __rpow__(self, other):
        return tf.pow(other, self)

    # logical operations
    def __invert__(self):
        return tf.logical_not(self)

    def __and__(self, other):
        return tf.logical_and(self, other)

    def __rand__(self, other):
        return tf.logical_and(other, self)

    def __or__(self, other):
        return tf.logical_or(self, other)

    def __ror__(self, other):
        return tf.logical_or(other, self)

    def __xor__(self, other):
        return tf.logical_xor(self, other)

    def __rxor__(self, other):
        return tf.logical_xor(other, self)

    # boolean operations
    def __lt__(self, other):
        return tf.less(self, other)

    def __le__(self, other):
        return tf.less_equal(self, other)

    def __gt__(self, other):
        return tf.greater(self, other)

    def __ge__(self, other):
        return tf.greater_equal(self, other)

    # slicing and indexing
    def __getitem__(self, item):
        return (tf.convert_to_tensor(self))[item]


def register_tensor_wrapper_class(cls):
    """Register the specified TensorWrapper type into TensorFlow type system.

    Parameters
    ----------
    cls : type
        A subclass of `TensorWrapper`, which is to be registered.
    """
    if not isinstance(cls, six.class_types) or \
            not issubclass(cls, TensorWrapper):
        raise TypeError('`%s` is not a type, or not a subclass of '
                        '`TensorWrapper`.' % (cls,))

    def to_tensor(value, dtype=None, name=None, as_ref=False):
        if dtype and not dtype.is_compatible_with(value.dtype):
            raise ValueError('Incompatible type conversion requested to type '
                             '%s for tensor of type %s' %
                             (dtype.name, value.dtype.name))
        if as_ref:
            raise ValueError('%r: Ref type not supported.' % value)
        return value.__wrapped__

    tf.register_tensor_conversion_function(cls, to_tensor)

    # bring support for session.run(StochasticTensor), and for using as keys
    # in feed_dict.
    register_session_run_conversion_functions(
        cls,
        fetch_function=lambda t: ([getattr(t, '__wrapped__')],
                                  lambda val: val[0]),
        feed_function=lambda t, v: [(getattr(t, '__wrapped__'), v)],
        feed_function_for_partial_run=lambda t: [getattr(t, '__wrapped__')]
    )
