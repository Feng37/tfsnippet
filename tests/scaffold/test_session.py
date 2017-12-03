import pytest
import tensorflow as tf

from tfsnippet.scaffold import (get_default_session_or_error,
                                get_variables_as_dict, VariableSaver)
from tfsnippet.utils import TemporaryDirectory


class GetDefaultSessionOrErrorTestCase(tf.test.TestCase):

    def test_get_default_session_or_error(self):
        with pytest.raises(RuntimeError, message='No session is active'):
            get_default_session_or_error()
        with self.test_session(use_gpu=False) as sess:
            self.assertIs(sess, get_default_session_or_error())
        with pytest.raises(RuntimeError, message='No session is active'):
            get_default_session_or_error()


class GetVariablesTestCase(tf.test.TestCase):

    def test_get_variables_as_dict(self):
        GLOBAL_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
        MODEL_VARIABLES = tf.GraphKeys.MODEL_VARIABLES
        LOCAL_VARIABLES = tf.GraphKeys.LOCAL_VARIABLES

        # create the variables to be checked
        a = tf.get_variable(
            'a', shape=(), collections=[GLOBAL_VARIABLES, MODEL_VARIABLES])
        b = tf.get_variable(
            'b', shape=(), collections=[GLOBAL_VARIABLES])
        c = tf.get_variable(
            'c', shape=(), collections=[MODEL_VARIABLES])

        with tf.variable_scope('child') as child:
            child_a = tf.get_variable(
                'a', shape=(),
                collections=[GLOBAL_VARIABLES, MODEL_VARIABLES])
            child_b = tf.get_variable(
                'b', shape=(), collections=[GLOBAL_VARIABLES])
            child_c = tf.get_variable(
                'c', shape=(), collections=[MODEL_VARIABLES])

        # test to get variables as dict
        self.assertEqual(
            get_variables_as_dict(),
            {'a': a, 'b': b, 'child/a': child_a, 'child/b': child_b}
        )
        self.assertEqual(
            get_variables_as_dict(collection=MODEL_VARIABLES),
            {'a': a, 'c': c, 'child/a': child_a, 'child/c': child_c}
        )
        self.assertEqual(
            get_variables_as_dict(collection=LOCAL_VARIABLES),
            {}
        )
        self.assertEqual(
            get_variables_as_dict(''),
            {'a': a, 'b': b, 'child/a': child_a, 'child/b': child_b}
        )
        self.assertEqual(
            get_variables_as_dict('child'),
            {'a': child_a, 'b': child_b}
        )
        self.assertEqual(
            get_variables_as_dict('child/'),
            {'a': child_a, 'b': child_b}
        )
        self.assertEqual(
            get_variables_as_dict(child),
            {'a': child_a, 'b': child_b}
        )
        self.assertEqual(
            get_variables_as_dict('child', collection=MODEL_VARIABLES),
            {'a': child_a, 'c': child_c}
        )
        self.assertEqual(
            get_variables_as_dict('child', collection=LOCAL_VARIABLES),
            {}
        )
        self.assertEqual(
            get_variables_as_dict('non_exist'),
            {}
        )


class VariablesSaverTestCase(tf.test.TestCase):

    def test_save_restore(self):
        a = tf.get_variable('a', initializer=1, dtype=tf.int32)
        b = tf.get_variable('b', initializer=2, dtype=tf.int32)
        c = tf.get_variable('c', initializer=3, dtype=tf.int32)
        a_ph = tf.placeholder(dtype=tf.int32, shape=(), name='a_ph')
        b_ph = tf.placeholder(dtype=tf.int32, shape=(), name='b_ph')
        c_ph = tf.placeholder(dtype=tf.int32, shape=(), name='c_ph')
        assign_op = tf.group(
            tf.assign(a, a_ph),
            tf.assign(b, b_ph),
            tf.assign(c, c_ph)
        )

        def get_values(sess):
            return sess.run([a, b, c])

        def set_values(sess, a, b, c):
            sess.run(assign_op, feed_dict={a_ph: a, b_ph: b, c_ph: c})

        with TemporaryDirectory() as tempdir1, \
                TemporaryDirectory() as tempdir2:
            saver1 = VariableSaver([a, b, c], tempdir1)
            saver2 = VariableSaver({'aa': a, 'bb': b}, tempdir2)

            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                self.assertEqual(get_values(sess), [1, 2, 3])
                set_values(sess, 10, 20, 30)
                self.assertEqual(get_values(sess), [10, 20, 30])

                saver1.save()
                set_values(sess, 100, 200, 300)
                self.assertEqual(get_values(sess), [100, 200, 300])
                saver1.restore()
                self.assertEqual(get_values(sess), [10, 20, 30])

                saver2.save()
                set_values(sess, 100, 200, 300)
                self.assertEqual(get_values(sess), [100, 200, 300])
                saver2.restore()
                self.assertEqual(get_values(sess), [10, 20, 300])

                saver1.restore()
                self.assertEqual(get_values(sess), [10, 20, 30])

                set_values(sess, 101, 201, 301)
                saver2.save()
                set_values(sess, 100, 200, 300)
                self.assertEqual(get_values(sess), [100, 200, 300])
                saver2.restore()
                self.assertEqual(get_values(sess), [101, 201, 300])

    def test_non_exist(self):
        with TemporaryDirectory() as tempdir:
            a = tf.get_variable('a', initializer=1, dtype=tf.int32)
            saver = VariableSaver([a], tempdir)
            with pytest.raises(
                    IOError,
                    match='Checkpoint file does not exist in directory'):
                saver.restore()
            saver.restore(ignore_non_exist=True)

    def test_errors(self):
        with TemporaryDirectory() as tempdir:
            a = tf.get_variable('a', initializer=1, dtype=tf.int32)
            with pytest.raises(
                    ValueError, message='At least 2 versions should be kept'):
                _ = VariableSaver([a], tempdir, max_versions=1)
