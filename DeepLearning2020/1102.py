import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

MSE = tf.keras.losses.MeanSquaredError()
def mse_loss():
     y = a*x**2 + b*x+c
##     y = a*tf.pow(x, 2) + b**x + c
     return MSE(y, t) # tf.reduce_mean(tf.square(y - t))

EPOCH = 1000
train_size = 20

# create the train data
tf.random.set_seed(1) # np.random.seed(1)
x = tf.linspace(-5.0, 5.0, num=train_size)

a_true = tf.Variable(3.0)
b_true = tf.Variable(2.0)
c_true = tf.Variable(1.0)
t = a_true*tf.pow(x, 2) + b_true*x+c_true
t += tf.random.normal([train_size], mean=0.0, stddev = 2)
#t = tf.add(t, np.random.normal(0, 2.0, train_size))

a = tf.Variable(tf.random.normal([]))
b = tf.Variable(tf.random.normal([]))
c = tf.Variable(tf.random.normal([]))
                
opt = tf.keras.optimizers.SGD(learning_rate=0.001)
#opt = tf.keras.optimizers.Adam(learning_rate=0.01)
##opt = tf.keras.optimizers.RMSprop(0.01)

loss_list = [ ]
for epoch in range(EPOCH):   
    opt.minimize(mse_loss, var_list= [a, b, c])
     
    loss = mse_loss().numpy()
    loss_list.append(loss)

    if not epoch%100:
        print("epoch={}: loss={}".format(epoch, loss))      

print("a={:>.4f}. b={:>.4f}, c={:>.4f}, loss={}".format(
       a.numpy(), b.numpy(), c.numpy(),loss))

plt.plot(loss_list)
plt.show()

plt.scatter(x, t.numpy())

t_pred = a*tf.pow(x, 2) + b*x + c # parabola curve
plt.plot(x, t_pred, 'red')
plt.show()
