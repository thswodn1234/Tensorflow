import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

tf.random.set_seed(1)
W = tf.Variable(tf.random.normal(shape=[2, 1]), )
b = tf.Variable(tf.random.normal(shape=[1]))
lr = 0.01   # learning rate, 0.001

train_size = X.shape[0]
batch_size = 4
K = train_size// batch_size

loss_list = [ ]
for epoch in range(1000):
    batch_loss = 0.0
    for step in range(K):
        mask = np.random.choice(train_size, batch_size)
        x_batch = X[mask]
        t_batch = t[mask]
        
        with tf.GradientTape() as tape:
            y = tf.matmul(x_batch, W) + b
            loss = MSE(y, t_batch)
            
        batch_loss += loss.numpy()
        
        dW, dB = tape.gradient(loss, [W, b])
        W.assign_sub(lr * dW)
        b.assign_sub(lr * dB)
        
    batch_loss /= K    
    loss_list.append(batch_loss) # average loss    
##    if not epoch%100:
##            print("epoch={}, batch_loss={}".format(epoch, batch_loss))

print("W={}. b={}, loss={}".format(W.numpy(), b.numpy(), batch_loss))

plt.plot(loss_list)
plt.show()
