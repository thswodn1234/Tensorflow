import tensorflow as tf
import numpy as np

#1
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: crate a 1D input data
A = np.array([1, 2, 3, 4, 5]).astype('float32')

#3: calculate output size in padding="valid"
k = 2      # pool_size, kernel_size
s = 2      # slides
n = len(A) # input_size
# the same as pad1d_infor(padding="valid")
new_steps= int(np.ceil((n - k + 1) / s))
print("new_steps = ", new_steps) # len(B)

#4: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(5, 1)))
model.add(tf.keras.layers.AveragePooling1D(pool_size=k, strides =s)) 
model.summary()

#5: apply A to model
A = np.reshape(A, (1, 5, 1)) # (batch, steps, channels)
output = model.predict(A)    # (batch, downsampled_steps, channels)
B = output.flatten()
print("B=", B)
