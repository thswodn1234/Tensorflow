import tensorflow as tf
import numpy as np

#1
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: crate a 1D input data
A = np.array([1, 2, 3, 4, 5]).astype('float32')
A = np.reshape(A, (1, 5, 1))  #  (batch, steps, channels)

#3: build a model
KERNEL_SIZE = 3 
STRIDE      = 1 
PADDING  = 'valid'  # 'same’

model = tf.keras.Sequential()
##model.add(tf.keras.layers.Input(shape = A.shape[1:])) # (5, 1)
##model.add(tf.keras.layers.Conv1D(filters=1,
##                                 kernel_size = KERNEL_SIZE,
##                                 strides= STRIDE,
##                                 padding= PADDING,
##                                 use_bias=False,
##                                 kernel_initializer=tf.constant_initializer(1),
##                                 ))
model.add(tf.keras.layers.Conv1D(filters=1,
                                 kernel_size = KERNEL_SIZE,
                                 strides= STRIDE,
                                 padding= PADDING,
                                 use_bias=False,
                                 kernel_initializer=tf.constant_initializer(1),
                                 input_shape=A.shape[1:])) # (5, 1)
model.summary()

#4: apply A to model 
output = model.predict(A) # output.shape : (batch, new_steps, filters)
B = output.flatten()       # B = tf.reshape(output,[-1]).numpy()
print("B=", B)
