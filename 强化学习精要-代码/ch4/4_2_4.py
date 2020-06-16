# import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorflow as tf_ori
tf.disable_v2_behavior()

a = tf.constant(1.0)
b = tf.constant(1.0)
c = a + b
g = tf.get_default_graph()
print( g.get_operations())
grad = tf.gradients(c, [a,b])
print( g.get_operations())
tf.summary.FileWriter("logs", g).close()