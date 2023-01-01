import tensorflow as tf

x = tf.Variable(3.0)

with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = x**3
    dy = tape1.gradient(y, x)
dy2 = tape2.gradient(dy, x)
print('dy=', dy.numpy())
print('dy2=', dy2.numpy())
