# import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorflow as tf_ori
tf.disable_v2_behavior()

a = tf.Variable(1, name='a')
g = tf.get_default_graph()
print( g.get_operations())
