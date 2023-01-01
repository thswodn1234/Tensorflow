import tensorflow as tf

x = tf.constant(2.0)
y = tf.constant(3.0)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    tape.watch(y)
    z = x**2 + y**2
dx = tape.gradient(z, x)
dy = tape.gradient(z, y)

print('dx=', dx.numpy())
print('dy=', dy.numpy())
