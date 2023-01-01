import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import numpy as np
import matplotlib.pyplot as plt

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: 
#np.random.seed(1)
#X = np.arange(50)
#np.random.shuffle(X)
#X = X.reshape(-1, 10)
X = np.array([[27, 35, 40, 38,  2,  3, 48, 29, 46, 31],
              [32, 39, 21, 36, 19, 42, 49, 26, 22, 13],
              [41, 17, 45, 24, 23,  4, 33, 14, 30, 10],
              [28, 44, 34, 18, 20, 25,  6,  7, 47,  1],
              [16,  0, 15,  5, 11,  9,  8, 12, 43, 37]], dtype=np.float)
# normalize 
##A = X/np.max(X)
mX = np.mean(X, axis = 0)
std = np.std(X, axis = 0)
A = (X - mX)/std

#3: autoencoder model
#3-1:
encode_dim = 4  # latent_dim
input_x = Input(shape = (10,))  #  A.shape[1:]
encode= Dense(units = 8, activation = 'relu')(input_x)
encode= Dense(units = encode_dim, activation = 'relu')(encode)
encoder = tf.keras.Model(inputs= input_x, outputs= encode)
encoder.summary()

#3-2:
decode_input = Input(shape = (encode_dim,))
decode= Dense(units = 8, activation = 'relu')(decode_input)
decode= Dense(units = 10, activation = None)(decode)
decoder = tf.keras.Model(inputs= decode_input, outputs= decode)
decoder.summary

#3-3:
autoencoder  = tf.keras.Model(inputs = input_x,  outputs = decoder(encoder(input_x)))
autoencoder.summary()
 
#4: train the model
opt = tf.keras.optimizers.RMSprop(learning_rate = 0.001) #'rmsprop'
autoencoder.compile(optimizer = opt, loss= 'mse', metrics = ['accuracy'])
ret = autoencoder.fit(A, A, epochs = 2000, batch_size= 3, verbose=0)

#5:
x = encoder(A)
print("x=\n", x)

B = decoder(x)  # B = autoencoder(A), hat(X) = B*std + mX
##print("B=\n", B)
##print("A=\n", A)  # input
print("np.abs(A - B)=\n", np.abs(A - B))

#6:
fig, ax = plt.subplots(1, 2, figsize = (10, 6))
ax[0].plot(ret.history['loss'], "g-")
ax[0].set_title("train loss")
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

ax[1].plot(ret.history['accuracy'], "b-")
ax[1].set_title("accuracy")
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')
fig.tight_layout()
plt.show()
