import tensorflow as tf

a= tf.fill([2, 3], 2.0)
print(a)

b = tf.linspace(0.0, 1.0, 5)
print(b)

c = tf.range(5)
print(c)

d = tf.range(1, 5, 0.5)
print(d)

w = tf.Variable(d)
print(w)
