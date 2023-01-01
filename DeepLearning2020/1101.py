import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

MSE = tf.keras.losses.MeanSquaredError()
def mse_loss():
     y = x*w + b
     return MSE(y, t) # tf.reduce_mean(tf.square(y - t))

EPOCH = 1000
train_size = 20

# create the train data
tf.random.set_seed(1) # np.random.seed(1)
x = tf.linspace(0.0, 10.0, num=train_size) #np.linspace(0.0, 10.0, num=20)
w_true, b_true = 3, -10  # truth, line parameters
t = x*w_true + b_true + tf.random.normal([train_size], mean=0.0, stddev=2.0)

# train parameters
w = tf.Variable(tf.random.normal([]))
b = tf.Variable(tf.random.normal([]))

opt = tf.keras.optimizers.SGD(learning_rate=0.01)

loss_list = [ ]
for epoch in range(EPOCH):   
    opt.minimize(mse_loss, var_list= [w, b])
     
    loss = mse_loss().numpy()
    loss_list.append(loss)
    if not epoch%100:
        print("epoch={}: loss={}".format(epoch, loss))

print("w={:>.4f}. b={:>.4f}, loss={}".format(w.numpy(), b.numpy(), loss))

plt.plot(loss_list)
plt.show()

plt.scatter(x, t.numpy())  # train data plot

w_pred, b_pred = w.numpy(), b.numpy() # predicted, line parameters
t_pred= x*w_pred + b_pred 
plt.plot(x, t_pred, 'r-')
plt.show()
