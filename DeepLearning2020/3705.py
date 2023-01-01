import tensorflow as tf
import numpy as np

#1
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: crate a 1D input data with 3-channels
A = np.array([[1, 1, 1],
              [2, 2, 2],
              [3, 3, 3],
              [4, 4, 4],
              [5, 5, 5]], dtype='float32')
A = np.expand_dims(A, axis=0) # shape = ([1, 5, 3])

#3: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape = (5, 3)))
model.add(tf.keras.layers.MaxPool1D()) #pool_size = 2, strides =2
##model.add(tf.keras.layers.AveragePooling1D())
##model.add(tf.keras.layers.MaxPool1D(padding='same')
##model.add(tf.keras.layers.AveragePooling1D(padding='same'))
##model.summary()

#4: apply A to model
B = model.predict(A) # (batch, downsampled_steps, channels)
print("B=", B)
print("B[:,:,0]=", B[:,:,0]) # 0-channel
print("B[:,:,1]=", B[:,:,1]) # 1-channel
print("B[:,:,2]=", B[:,:,2]) # 2-channel
