from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf

print(tf.executing_eagerly())
disable_eager_execution()
print(tf.executing_eagerly())

# Graph construction
a = tf.constant(1)
b = tf.constant(2)
c = a + b  # c= tf.add(a, b)
print(a)
print(b)
print(c)
 

#2 graph execution
sess = tf.compat.v1.Session()
print(sess.run(a))
print(sess.run(b))
print(sess.run(c))
sess.close()
