import tensorflow as tf
import numpy as np

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: crate a 2D input data
A = np.array([[1, 2, 3, 4, 5],
              [4, 3, 2, 1, 0],
              [5, 6, 7, 8, 9],
              [4, 3, 2, 1, 0],
              [0, 1, 2, 3, 4]],dtype='float32')
A = A.reshape(-1, 5, 5, 1)

PADDING  =  'valid' #  'same'

#3: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(A.shape[1:])) # shape=(5, 5, 1)
model.add(tf.keras.layers.Conv2D(filters=1,
                                 kernel_size = (2, 2),
                                 strides= (2, 2),
                                 padding= PADDING,
                                 use_bias=False,
                                 kernel_initializer=tf.constant_initializer(1/4),
                                 input_shape=A.shape[1:])) # (5, 5, 1)
model.summary()

#4: apply A to model
B = model.predict(A)  # (batch, new_rows, new_cols, filters)
##output = model(A); B = output.numpy()
print("B=", B)

#5: weights
W = model.trainable_variables[0] # (kernel_size[0], kernel_size[1], channels, filters)
print("W.shape=", W.shape)
print("W[:, :, 0, 0]=", W[:, :, 0, 0])
