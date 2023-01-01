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
A = A.reshape(-1, 5, 5, 1)  # (batch, rows, cols, channels)

#3: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(A.shape[1:])) # shape=(5, 5, 1)
model.add(tf.keras.layers.MaxPool2D())
##model.add(tf.keras.layers.MaxPool2D(padding='same'))
model.summary()

#4: apply A to model
B = model.predict(A)  # (batch, pooled_rows, pooled_cols, channels)
print("B=", B)
