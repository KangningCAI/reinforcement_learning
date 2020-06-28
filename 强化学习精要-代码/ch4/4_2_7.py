# import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorflow as tf_ori
tf.disable_v2_behavior()

add_arg_scope = tf_ori.contrib.framework.add_arg_scope
arg_scope = tf_ori.contrib.framework.arg_scope


@add_arg_scope
def func1(*args, **kwargs):
    return (args, kwargs)

with arg_scope((func1,), a=1, b=None, c=[1]):
    args, kwargs = func1(0)
    print( args)
    print( kwargs)