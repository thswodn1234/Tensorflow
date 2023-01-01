import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = np.arange(12)
t = np.arange(12)
#x = tf.convert_to_tensor(x, dtype=tf.float32)
#t = tf.convert_to_tensor(t, dtype=tf.float32)

w = tf.Variable(0.5)
b = tf.Variable(0.0)
lr = 0.001   # learning rate

loss_list = [ ]  # for graph 
for epoch in range(100):   
    with tf.GradientTape() as tape:
        y = x*w + b
        loss = tf.reduce_mean(tf.square(y - t))
    loss_list.append(loss.numpy())

    dW, dB = tape.gradient(loss, [w, b])
    w.assign_sub(lr * dW) 
    b.assign_sub(lr * dB)
##    if not epoch%10:
##        print("epoch={}: w={:>.4f}. b={:>.4f}, loss={}".format(
##               epoch, w.numpy(), b.numpy(), loss.numpy()))

print("w={:>.4f}. b={:>.4f}, loss={}".format(w.numpy(), b.numpy(), loss.numpy()))

plt.plot(loss_list)
plt.show()
