import tensorflow as tf
import numpy as np

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
 
#2: crate a 2D input data
A = np.array([[1, 2, 3],
              [4, 5, 6]],dtype = 'float32')
A = A.reshape(-1, 2, 3, 1)     # (batch, rows, cols, channels)

#3: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(A.shape[1:])) # shape=(2, 3, 1)
#3-1
model.add(tf.keras.layers.UpSampling2D())     # size =(2,2), interpolation='nearest'
#3-2
##model.add(tf.keras.layers.UpSampling2D(interpolation= 'bilinear')) # size =(2,2)
model.summary()

#4: apply A to model
B = model.predict(A)      # (batch_size, upsampled_rows, upsampled_cols, channels)
print("B.shape=", B.shape)
print("B[0,:,:,0]=", B[0,:,:,0]) 
