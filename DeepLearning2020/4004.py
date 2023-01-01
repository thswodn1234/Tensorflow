import tensorflow as tf
import numpy as np

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: crate a 2D input data with 2-channels
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

print("A.shape", A.shape)       # (2, 5, 5)
A1 = np.transpose(A, (1, 2, 0)) # (5, 5, 2)
A1 = np.expand_dims(A1, axis=0) # (1, 5, 5, 2)

#3: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(A1.shape[1:])) # shape=(5, 5, 2)
model.add(tf.keras.layers.GlobalMaxPooling2D())
##model.add(tf.keras.layers.GlobalAveragePooling2D())
model.summary()

#4: apply A1 to model
B = model.predict(A1) # (batch, channels) = (1, 2)
##output = model(A1); B = output.numpy()
print("B=", B)
