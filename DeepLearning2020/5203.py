import tensorflow as tf
import numpy as np

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: crate a 2D input with 2 channels
A = np.array([[[1, 2],           # 0-channel
               [3, 4]],
              [[1, 2],           # 1-channel 
               [3, 4]]], dtype = 'float32')
##print("A.shape", A.shape)         # (channels, rows, cols) = (2, 2, 2)
A = np.transpose(A, (1, 2, 0))      # (rows, cols, channels) = (2, 2, 2)
A= np.expand_dims(A, axis = 0)      # (batch,rows, cols, channels)=(1, 2, 2, 2)

#3: kernel with 2-channels
W = np.array([[[1, -1],           # 0-channel
               [2, -2]],
              [[1, -1],           # 1-channel 
               [2, -2]]], dtype = 'float32')
##print("W.shape", W.shape)         # (channels, rows, cols) = (2, 2, 2)
W = np.transpose(W, (1, 2, 0))      # (rows, cols, channels) = (2, 2, 2)
W= np.expand_dims(W, axis = 2)      # (rows, cols, filters, channels) = (2, 2, 1, 2)

#4: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(A.shape[1:])) # shape = (2, 2, 2)
model.add(tf.keras.layers.Conv2DTranspose(filters=1,
                                 kernel_size = (2, 2),
                                 strides = (2, 2),
                                 padding = 'valid', # 'same          
                                 use_bias = False,
                                 kernel_initializer = tf.constant_initializer(W)))
model.summary()

#5: apply A to model
B = model.predict(A)     # (batch, new_rows, new_cols, filters)
print("B.shape=", B.shape)
print("B[0,:,:,0]=\n", B[0,:,:,0])
