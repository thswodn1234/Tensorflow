import tensorflow as tf
import numpy as np

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: 2D input data: A, B, C
A = np.array([[1, 2],
              [3, 4]],dtype='float32')
A = A.reshape(-1, 2, 2, 1) # (batch, rows, cols, channels)

B = np.array([[5, 6],
              [7, 8]],dtype='float32')
B = B.reshape(-1, 2, 2, 1) # (batch, rows, cols, channels)

C = np.array([1, 2, 3]).astype('float32')
C = C.reshape(-1, 3, 1, 1) # (batch, rows, cols, channels)
 
#3: build a model
x = tf.keras.layers.Input(shape=A.shape[1:]) # shape=(2, 2, 1)
y = tf.keras.layers.Input(shape=B.shape[1:]) # shape=(2, 2, 1)
z = tf.keras.layers.Input(shape=C.shape[1:]) # shape=(3, 1, 1) 
out3 = tf.keras.layers.Add()([x, y])
##out3 = tf.keras.layers.Subtract()([x, y])
##out3 = tf.keras.layers.Multiply()([x, y])
##out3 = tf.keras.layers.Minimum()([x, y])
##out3 = tf.keras.layers.Maximum()([x, y])
##out3 = tf.keras.layers.Average()([x, y])
out4 = tf.keras.layers.Concatenate()([x, y])
out5 = tf.keras.layers.Dot(axes= -1)([x, y])   # outer product
out6 = tf.keras.layers.Dot(axes= -1)([x, z])   # outer product
out_list = [x, y, z, out3, out4, out5, out6]
model = tf.keras.Model(inputs=[x, y, z], outputs= out_list)
##model.summary()
print("model.output_shape=", model.output_shape)

#4: apply [A, B, C] to model
##output = model([A, B, C])       # Tensor output
output = model.predict([A, B, C]) # numpy output
for i in range(len(output)):
    print("output[{}]={}".format(i, output[i]))   
