# import tensorflow as tf
import sys
import tensorflow.compat.v1 as tf
import tensorflow as tf_ori
tf.disable_v2_behavior()

def test1():
    with tf.name_scope('123'):
      with tf.name_scope('456'):
        with tf.variable_scope('789'):
            a = tf.Variable(1,name='a')
            print( a.name)
            b = tf.get_variable('b',1)
            print( b.name)

def test2():
    with tf.name_scope('123'):
      with tf.name_scope('456'):
        with tf.variable_scope('789'):
            a = tf.Variable(1,name='a')
            print( a.name)
            b = tf.get_variable('b',1)
            print( b.name)
      with tf.name_scope('456'):
        with tf.variable_scope('789'):
            c = tf.Variable(1,name='c')
            print( c.name)
            d = tf.Variable(1,name='d')
            print( d.name)


def test3():
    with tf.name_scope('123'):
        with tf.name_scope('456'):
            with tf.variable_scope('789'):
                d = tf.Variable(1,name='d')
                print( d.name)
            with tf.variable_scope('789'):
                e = tf.get_variable('e',1)
                print( e.name)

def test4():
    with tf.name_scope('123'):
        with tf.name_scope(None):
            c = tf.Variable(1,name='f')
            print( c.name)

if __name__ == '__main__':
    if sys.argv[1] == '1':
        test1()
    elif sys.argv[1] == '2':
        test2()
    elif sys.argv[1] == '3':
        test3()
    elif sys.argv[1] == '4':
        test4()


