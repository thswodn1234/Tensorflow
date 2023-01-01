import tensorflow as tf

tf.random.set_seed(1)
a = tf.range(5)
print(tf.random.shuffle(a))
print(tf.random.uniform(shape=(2, 3), minval=0, maxval=1))
print(tf.random.normal(shape=(2,3))) # mean=0, stddev=1
print(tf.random.normal(shape=(2,3), mean=10, stddev=2))
print(tf.random.truncated_normal(shape=(2, 3))) # mean=0, stddev=1

w= tf.Variable(tf.random.truncated_normal(shape=(2, 3)))
print(w)
