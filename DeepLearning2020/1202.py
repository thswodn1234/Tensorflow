import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_data = np.array([ # t = 1*x0 + 2*x1 + 3
#  x0, x1, t      
 [ 1,  0,  4],
 [ 2,  0,  5],
 [ 3,  0,  6],
 [ 4,  0,  7],
 [ 1,  1,  6],
 [ 2,  1,  7],
 [ 3,  1,  8],
 [ 4,  1,  9]], dtype=np.float32)

X      = train_data[:, :-1]
y_true = train_data[:, -1:]  # t
##y_true += np.reshape(np.random.randn(len(y_true))*2.0, (-1, 1)) 

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=2)) # input_shape=(2,)
model.summary()

##opt = tf.keras.optimizers.SGD(learning_rate=0.01)
##opt = tf.keras.optimizers.Adam(learning_rate=0.1)
opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
model.compile(optimizer=opt, loss='mse') # 'mean_squared_error'
##model.compile(optimizer='sgd', loss='mse') # 'sgd', 'adam', 'rmsprop'

# 0: silent, 1:progress bar,  2: one line per epoch 
ret = model.fit(X, y_true, epochs=100, batch_size=4, verbose=2)
y_pred = model.predict(X)
print("y_pred:", y_pred)
print("len(model.layers):", len(model.layers)) # 1

loss = ret.history['loss']
print("loss:", loss[-1])
#print(model.get_weights())
print("weights:", model.layers[0].weights[0].numpy())
print("bias:", model.layers[0].weights[1].numpy()) # model.layers[0].bias.numpy()

plt.plot(loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
