# import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorflow as tf_ori
tf.disable_v2_behavior()

a = tf.get_variable('a',1)
b = tf.get_variable('b',2)
c = a + b
g = tf.get_default_graph()
print( g.get_operations()[-1])