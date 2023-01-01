import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

##def dataset(train_size=100): # numpy
##     np.random.seed(1)
##     x = np.linspace(0.0, 10.0, num=train_size)
####     y = x**3 + x**2 + x + 4.0
##     y = 3.0*x - 10.0
####     y+= np.random.randn(train_size)*2.0
##     y += np.random.normal(loc=0.0, scale=2.0, size=train_size)
##     return x, y

def dataset(train_size=100): # tensorflow    
     tf.random.set_seed(1)
     x = tf.linspace(0.0, 10.0, num=train_size)
##     y = x**3 + x**2 + x + 4.0
     y = 3.0*x - 10.0
     y += tf.random.normal([train_size], mean=0.0, stddev = 2.0)
     return x, y
x, y_true = dataset(20)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=1))
##model.add(tf.keras.layers.Dense(units=1, input_shape=(1,))) # [1]
##model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=(1,))])
model.summary()

##opt = tf.keras.optimizers.SGD(learning_rate=0.01)
##opt = tf.keras.optimizers.Adam(learning_rate=0.1)
opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
model.compile(optimizer=opt, loss='mse') # 'mean_squared_error'
##model.compile(optimizer='sgd', loss='mse') # 'sgd', 'adam', 'rmsprop'

# 0: silent, 1:progress bar,  2: one line per epoch 
ret = model.fit(x, y_true, epochs=100, batch_size=4, verbose=2)
print("len(model.layers):", len(model.layers)) # 1

loss = ret.history['loss']
print("loss:", loss[-1])
#print(model.get_weights())  # weights, bias
print("weights:", model.layers[0].weights[0].numpy())
print("bias:", model.layers[0].weights[1].numpy()) # model.layers[0].bias.numpy()

plt.plot(loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.scatter(x, y_true)
y_pred = model.predict(x)
plt.plot(x, y_pred, color='red')
plt.show()
