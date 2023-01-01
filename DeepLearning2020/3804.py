import tensorflow as tf
import numpy as np

#1
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: crate a 1D input data
A = np.array([[1, 1, 1],
              [2, 1, 2],
              [3, 1, 0],
              [4, 1, 1],
              [5, 1, 2]], dtype='float32')
n = 2 # number of filters in Conv1D
steps = 5 # A.shape[0], 5, length

#3: kernel initial values, channels
W0 = np.ones(shape=(5, 3), dtype='float32')
W1 = np.full(shape=(5, 3), fill_value=2.0, dtype='float32')
W = np.stack((W0, W1), axis=2) # (5, 3, 2)

#4: Conv1D with n filters,  kernel_size =steps, strides = 1,
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(steps, 3)))
model.add(tf.keras.layers.Conv1D(filters= n, kernel_size= steps, use_bias= False,
                                 kernel_initializer= tf.constant_initializer(W))) 
model.add(tf.keras.layers.Flatten()) # output.shape : (batch, new_steps*filters)
model.summary()

#5: apply A to model
A = np.expand_dims(A, axis=0) # tf.expand_dims(A, axis=0), shape = ([1, 5, 3])
print("A = ", A)

output = model.predict(A) # output.shape =(1, 2)
print("output=", output)

##w = model.trainable_variables[0].numpy()
##print("w[:,:,0]=", w[:,:,0]) # W[:,:,0]
##print("w[:,:,1]=", w[:,:,1]) # W[:,:,1]
