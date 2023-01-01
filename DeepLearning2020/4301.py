import tensorflow as tf
import numpy as np

#1: 
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)

#2: create 2D input data
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8 ]],dtype='float32')
A = A.reshape(1, 2, 4, 1)  # (batch, rows, cols, channels)

#3: build a model
x = tf.keras.layers.Input(shape=A.shape[1:])
y = tf.keras.layers.Reshape([4, 2, 1])(x)  # (1, 4, 2, 1)
z = tf.keras.layers.Permute([2, 1, 3])(x)

model = tf.keras.Model(inputs=x, outputs= [y, z])
model.summary()

#4: apply A to model
##output = model(A)    # Tensor output
output = model.predict(A)  # numpy output
print("A[0,:,:,0]=",A[0,:,:,0])
print("output[0]=", output[0][0,:,:,0])  # y
print("output[1]", output[1][0,:,:,0])   # z
