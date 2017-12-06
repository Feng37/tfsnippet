import pytest
import six
import tensorflow as tf

from tfsnippet.modules import ListMapper, DictMapper, Linear


class ListMapperTestCase(tf.test.TestCase):

    def test_outputs(self):
        net = ListMapper([
            Linear(2),
            (lambda x: x * tf.get_variable('y', initializer=0.)),
        ])
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        outputs = net(inputs)
        self.assertIsInstance(outputs, list)
        for v in outputs:
            self.assertIsInstance(v, tf.Tensor)

        _ = net(inputs)
        self.assertEqual(
            sorted(v.name for v in tf.global_variables()),
            ['linear/fully_connected/biases:0',
             'linear/fully_connected/weights:0',
             'list_mapper/_1/y:0']
        )

    def test_empty_mapper(self):
        with pytest.raises(ValueError, message='`mapper` must not be empty'):
            _ = ListMapper([])


class DictMapperTestCase(tf.test.TestCase):

    def test_outputs(self):
        net = DictMapper({
            'a': Linear(2),
            'b': lambda x: x * tf.get_variable('y', initializer=0.)
        })
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        output = net(inputs)
        self.assertIsInstance(output, dict)
        self.assertEqual(sorted(output.keys()), ['a', 'b'])
        for v in six.itervalues(output):
            self.assertIsInstance(v, tf.Tensor)

        _ = net(inputs)
        self.assertEqual(
            sorted(v.name for v in tf.global_variables()),
            ['dict_mapper/b/y:0',
             'linear/fully_connected/biases:0',
             'linear/fully_connected/weights:0']
        )

    def test_empty_mapper(self):
        with pytest.raises(ValueError, message='`mapper` must not be empty'):
            _ = DictMapper({})

    def test_invalid_key(self):
        for k in ['.', '', '90ab', 'abc.def']:
            with pytest.raises(
                ValueError, message='The key for `DictMapper` must be a valid '
                                    'Python identifier (matching the pattern '
                                    '"^[A-Za-z_][A-Za-z0-9_]*$")'):
                _ = DictMapper({k: lambda x: x})
