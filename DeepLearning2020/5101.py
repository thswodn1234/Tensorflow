import tensorflow as tf
import numpy as np

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: crate a 1D input data
A = np.array([1, 2, 3, 4, 5], dtype='float32')
 
#3: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape = (5, 1)))
model.add(tf.keras.layers.UpSampling1D())  # size = 2
##model.add(tf.keras.layers.Flatten())       # (batch, upsampled_steps*features)
model.summary()

#4: apply A to model
A = np.reshape(A, (1, 5, 1))  # (batch_size, steps, features)
output = model.predict(A)     # (batch_size, upsampled_steps, features)
B = output.flatten()
print("B=", B)
