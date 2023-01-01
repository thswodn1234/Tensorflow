import tensorflow as tf
import numpy as np

x = tf.constant([-10, -1.0, 0.0, 1.0, 10], dtype = tf.float32)

y1 = tf.keras.activations.linear(x) 
y2 = tf.keras.activations.sigmoid(x)
y3 = tf.keras.activations.tanh(x)
y4 = tf.keras.activations.relu(x)
y5 = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
y6 = tf.keras.activations.softmax(tf.reshape(x, shape=(1, -1)))

##linear = tf.keras.activations.get('linear')
##y1 = linear(x)
##
##sigmoid = tf.keras.activations.get('sigmoid')
##y2 = sigmoid(x)
##
##tanh = tf.keras.activations.get('tanh')
##y3 = tanh(x)
##
##relu = tf.keras.activations.get('relu')
##y4 = relu(x)
##
##y5 = relu(x, alpha=0.1) # LeakyReLU
##softmax = tf.keras.activations.get('softmax')
##y6 = softmax(tf.reshape(x, shape=(1, -1)))

print("y1=", y1.numpy())
print("y2=", y2.numpy())
print("y3=", y3.numpy())
print("y4=", y4.numpy())
print("y5=", y5.numpy())
print("y6=", y6.numpy())
print("sum(y6)=", np.sum(y6.numpy())) # 1.0
