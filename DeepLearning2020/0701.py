import tensorflow as tf

x = tf.Variable(2.0) # tf.Variable(2.0, trainable=True)
y = tf.Variable(3.0) # tf.Variable(3.0, trainable=True)

with tf.GradientTape() as tape:
    z = x**2 + y**2
dx, dy = tape.gradient(z, [x, y])

print('dx=', dx.numpy())
print('dy=', dy.numpy())
