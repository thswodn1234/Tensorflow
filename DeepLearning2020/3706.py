import tensorflow as tf
import numpy as np

#1
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)

#2: crate a 1D input data with 3-channels
A = np.array([[1, 0, 0],
              [2, 4, 0],
              [3, 3, 3],
              [4, 2, 2],
              [5, 1, 1]], dtype='float32')
A = np.expand_dims(A, axis=0) # (batch, steps, channels)= ([1, 5, 3])

#3: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(5, 3)))
##model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.summary()

#4: apply A to model
output = model.predict(A)   # (batch, channels) = (1, 3)
print("output=", output)
