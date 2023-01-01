import tensorflow as tf
import numpy as np

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: crate a 2D input data
A = np.array([[1, 2],
              [3, 4 ]],dtype='float32')
A = A.reshape(-1, 2, 2, 1)

#3: kernel
W = np.array([[ 1,  -1],      
              [ 2,  -2]], dtype = 'float32')
W = W.reshape(2, 2, 1, 1)   # (kernel_size[0], kernel_size[1], filters, channels)

#4: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(A.shape[1:])) # shape = (2, 2, 1)
model.add(tf.keras.layers.Conv2DTranspose(filters=1,
                                 kernel_size = (2, 2),
                                 strides = (2, 2),
                                 padding = 'valid',  # 'same'
                                 use_bias = False,
                                 kernel_initializer = tf.constant_initializer(W)))
model.summary()
##model.set_weights([W]) # kernel_initializer = tf.constant_initializer(W)

#5: apply A to model
B = model.predict(A)     # (batch, new_rows, new_cols, filters)
print("B.shape=", B.shape)
print("B[0,:,:,0]=\n", B[0,:,:,0])

#6: weights
##W1 = model.get_weights() # W, model.trainable_variables
##print("W1[0].shape=", W1[0].shape)
##print("W1[0]=\n", W1[0])
