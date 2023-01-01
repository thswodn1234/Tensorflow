'''
ref1:https://towardsdatascience.com/autoencoders-in-keras-c1f57b9a2fd7
ref2:
https://medium.com/analytics-vidhya/building-a-convolutional-autoencoder-using-keras-using-conv2dtranspose-ca403c8d144e
'''
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense,  Flatten, Reshape, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, UpSampling2D

import numpy as np
import matplotlib.pyplot as plt

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2:
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')/255
x_test  = x_test.astype('float32')/255

# expand data with channel = 1
x_train = np.expand_dims(x_train,axis = 3)     # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis = 3)      # (10000, 28, 28, 1)

#3: add noise to dataset
x_train_noise = x_train + np.random.normal(loc=0.0, scale=0.2, size=x_train.shape)
x_test_noise  = x_test  + np.random.normal(loc=0.0, scale=0.2, size=x_test.shape)
x_train_noise = np.clip(x_train_noise, 0, 1)
x_test_noise = np.clip(x_test_noise, 0, 1)

#4: autoencoder model
#4-1:
encode_dim = 32  # latent_dim
input_x= Input(shape = x_train.shape[1:]) #  (28, 28, 1)
encode= Conv2D(filters=32,kernel_size=(3,3), padding='same', activation = 'relu')(input_x)
encode= MaxPool2D()(encode) # (14, 14, 32)

encode= Conv2D(filters= 16, kernel_size= (3, 3), padding='same', activation = 'relu')(encode)
encode= MaxPool2D()(encode) # (7, 7, 16)
encode= Flatten()(encode)

encode= Dense(units = encode_dim, activation = 'relu')(encode)
encoder = tf.keras.Model(inputs= input_x, outputs= encode, name ='encoder')
encoder.summary()

#4-2: decoder by (Conv2D + UpSampling2D) or Conv2DTranspose
decode_input = Input(shape = (encode_dim,))
encode= Dense(units = 7*7*4, activation = 'relu')(decode_input)
decode= Reshape((7, 7, 4))(encode)

##decode=Conv2D(filters=16, kernel_size = (3, 3), strides = (1, 1), 
##                 activation = 'relu', padding = 'same')(decode)
##decode=UpSampling2D()(decode) # size =(2,2)
decode=Conv2DTranspose(filters=16, kernel_size = (3, 3), strides = (2, 2), 
                           activation = 'relu', padding = 'same')(decode)

##decode=Conv2D(filters=32, kernel_size = (3, 3), strides = (1, 1), 
##                 activation = 'relu', padding = 'same')(decode)
##decode=UpSampling2D()(decode)
decode=Conv2DTranspose(filters=32, kernel_size = (3, 3), strides = (2, 2), 
                           activation = 'relu', padding = 'same')(decode)

decode=Conv2D(filters=1, kernel_size = (3, 3), strides = (1, 1), 
              activation = 'sigmoid', padding = 'same')(decode)
decoder = tf.keras.Model(inputs= decode_input, outputs= decode, name ='decoder')
decoder.summary()

#4-3
autoencoder  = tf.keras.Model(inputs = input_x,
                            outputs = decoder(encoder(input_x)), name ='autoencoder')
autoencoder.summary()
 
#5: train the model
opt = tf.keras.optimizers.RMSprop(learning_rate = 0.001) #'rmsprop'
autoencoder.compile(optimizer = opt, loss= 'mse' ) #  'binary_crossentropy'
ret = autoencoder.fit(x=x_train_noise, y=x_train, epochs= 100, batch_size= 128,
                       validation_split = 0.2, verbose = 2) 

#6:
##fig, ax = plt.subplots(1, 2, figsize = (10, 6))
##ax[0].plot(ret.history['loss'], "b-")
##ax[0].set_title("train loss")
##ax[0].set_xlabel('epochs')
##ax[0].set_ylabel('loss')
##
##ax[1].plot(ret.history['val_loss'], "g-")
##ax[1].set_title("val loss")
##ax[1].set_xlabel('epochs')
##ax[1].set_ylabel('loss')
##fig.tight_layout()
##plt.show()

#7: apply  x_test_noise[:8] to model and display
F = encoder(x_test_noise[:8])
print("F.shape=", F.shape)

img = decoder(F)  # img = autoencoder(x_test_noise[:8])
img = img.numpy()
img = img.reshape(-1, 28, 28)
print("img.shape=", img.shape)
 
#8: display images
fig = plt.figure(figsize = (16, 4))
for i in range(16):   
    plt.subplot(2, 8, i + 1)
    if i<8: # noise
        plt.imshow(x_test_noise[i,:,:,0], cmap = 'gray')
    else:   # reconstructed
        plt.imshow(img[i-8], cmap = 'gray')
    plt.axis("off")

fig.tight_layout()
plt.show()
