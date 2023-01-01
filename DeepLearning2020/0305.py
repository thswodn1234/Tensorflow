import tensorflow as tf

a = tf.zeros(shape = (2, 3)) # dtype = tf.float32
print(a)

b = tf.ones(shape = (2, 3))
print(b)

c = tf.zeros_like(b)
print(c)

d = tf.ones_like(c)
print(d)

w = tf.Variable( d )
print(w)
