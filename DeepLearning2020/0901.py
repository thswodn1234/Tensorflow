import numpy as np
import tensorflow as tf

##def MSE(y, t):
##    return tf.reduce_mean(tf.square(y - t)) # (y - t)**2
MSE = tf.keras.losses.MeanSquaredError()

train_data = np.array([ # t = 1*x1 + 2*x2 + 3
#  x1, x2, t      
 [ 1,  0,  4],
 [ 2,  0,  5],
 [ 3,  0,  6],
 [ 4,  0,  7],
 [ 1,  1,  6],
 [ 2,  1,  7],
 [ 3,  1,  8],
 [ 4,  1,  9]], dtype=np.float32)

X = train_data[:, :-1]
t = train_data[:, -1:]  
#X = tf.convert_to_tensor(X, dtype=tf.float32)
#t = tf.convert_to_tensor(t, dtype=tf.float32)
print("X=", X)
print("t=", t)

tf.random.set_seed(1) # 난수열 초기화
W = tf.Variable(tf.random.normal(shape=[2, 1]), )
b = tf.Variable(tf.random.normal(shape=[1]))
##W = tf.Variable([[0.5],[0.5]], dtype=tf.float32)
##b = tf.Variable(0.0)
print("W=", W.numpy())
print("b=", b.numpy())

y = tf.matmul(X, W) + b
print("y=", y.numpy())

loss = MSE(y, t)
print("MSE(y, t)=", loss.numpy())
