import tensorflow as tf
import numpy as np

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: crate a 2D input data
A = np.array([[[1, 2, 3, 4, 5],  # 0-channel
               [4, 3, 2, 1, 0],
               [5, 6, 7, 8, 9],
               [4, 3, 2, 1, 0],
               [0, 1, 2, 3, 4]],
              [[1, 2, 3, 4, 5],  # 1-channel 
               [4, 3, 2, 1, 0],
               [5, 6, 7, 8, 9],
               [4, 3, 2, 1, 0],
               [0, 1, 2, 3, 4]]],  dtype='float32')

##print("A.shape", A.shape)       # (2, 5, 5)
A1 = np.transpose(A, (1, 2, 0)) # (5, 5, 2)
A1 = np.expand_dims(A1, axis=0) # (1, 5, 5, 2)

PADDING  = 'valid'   # 'same'

#3: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(A1.shape[1:])) # shape=(5, 5, 2)
model.add(tf.keras.layers.Conv2D(filters=1,
                                 kernel_size = (2, 2),
                                 strides= (2, 2),
                                 padding= PADDING,
                                 use_bias=False,
                                 kernel_initializer=tf.constant_initializer(1/4),
                                 input_shape=A.shape[1:])) # (5, 5, 2)
model.summary()

#4: apply A to model
B = model.predict(A1)  # (batch, new_rows, new_cols, filters)
##output = model(A1);B = output.numpy()
print("B=", B)

#5: weights
W = model.trainable_variables[0] # (kernel_size[0], kernel_size[1], channels, filters)
print("W.shape=", W.shape)
##print("W[:,:,0,0]=", W[:,:,0,0]) # 0-channel, 0-filter
##print("W[:,:,0,0]=", W[:,:,1,0]) # 1-channel, 0-filter
