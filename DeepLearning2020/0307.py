import tensorflow as tf
#1
a = tf.range(6)
print(a)

b = tf.reshape(a, shape=(2, 3)) # tf.reshape(a, shape=(-1, 3))
print(b)

c = tf.reshape(b, shape=(-1,))
print(c)

#2
d = tf.transpose(b) # tf.transpose(b, perm=[1, 0])
print(d)
