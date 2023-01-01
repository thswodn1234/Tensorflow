import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

MSE = tf.keras.losses.MeanSquaredError()
def mse_loss():
     y = tf.zeros_like(x)
     for i in range(W.shape[0]):
          y += W[i]*(x**(i+1))
     y += b # bias
     return MSE(y, t) # tf.reduce_mean(tf.square(y - t))

EPOCH = 5000
train_size = 20

# create the train data
tf.random.set_seed(1)
x = tf.linspace(-5.0, 5.0, num=train_size)

w_true = tf.Variable([1.0, 2.0, 3.0])
b_true = tf.Variable(4.0)    
t = w_true[2]*x**3 + w_true[1]*x**2 + w_true[0]*x + b_true      
t += tf.random.normal([train_size], mean=0.0, stddev = 30)

# train variables
n = 4 # n-th polynomial curve
W = tf.Variable(tf.random.normal([n]))
b = tf.Variable(tf.random.normal([])) 

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
##opt = tf.keras.optimizers.RMSprop(0.01)

loss_list = [ ]
for epoch in range(EPOCH): 
    opt.minimize(mse_loss, var_list= [W, b])
     
    loss = mse_loss().numpy()
    loss_list.append(loss)

    if not epoch%100:
        print("epoch={}: loss={}".format(epoch, loss))
        
print("W={}. b={}, loss={}".format(W.numpy(), b.numpy(),loss))
plt.plot(loss_list)
plt.show()

plt.scatter(x, t.numpy())

# polynomial curve
t_pred = tf.zeros_like(x)
for i in range(W.shape[0]): # n = W.shape[0] 
     t_pred += W[i]*(x**(i+1))
t_pred += b # bias
     
plt.plot(x, t_pred, 'red')
plt.show()
