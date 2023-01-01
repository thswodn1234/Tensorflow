import tensorflow as tf
import numpy as np

#1
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: crate a 1D input data
A = np.array([1, 2, 3, 4, 5]).astype('float32')

#3: calculate padding and output size(new_steps)
k = 2      # pool_size, kernel_size
s = 2      # slides
n = len(A) # input_size

# the same as pad1d_infor(padding="same")
new_steps = int(np.ceil(n/s))
print("new_steps  = ", new_steps ) # len(C)

pad_width	= max((new_steps - 1) * s + k - n, 0)
pad_left	= pad_width // 2
pad_right	= pad_width - pad_left
print("pad_left = %s, pad_right=%s"%(pad_left, pad_right))

paddings = np.array([[pad_left, pad_right]])
B = tf.pad(A, paddings) # 0-padding, but mode don't care, not used padding values
print("B=", B)

#4: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape = (5, 1)))
model.add(tf.keras.layers.AveragePooling1D(pool_size = k, strides =s, padding="same"))
model.summary()

#5: apply A to model
A = np.reshape(A, (1, 5, 1)) # (batch, steps, channels)
output = model.predict(A)    # (batch, downsampled_steps, channels)
C = output.flatten()
print("C=", C)
