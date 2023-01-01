import tensorflow as tf
import numpy as np
 
#1: crate a 2D input data
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]).astype('float32')
A = A.reshape(-1, 3, 3, 1)  # (batch, rows, cols, channels)
 
#2: build a model
pads = np.array([[1, 1],  # rows: (left, right) padding
                [2, 2]])  # cols: (top, bottom) padding 
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=A.shape[1:]))  # (3,3,1)
model.add(tf.keras.layers.ZeroPadding2D(padding=pads))
model.summary()

#3: apply A to model
output = model(A)
print("output.shape=", output.shape)
##print("output.numpy()=", output.numpy())

#4: apply A to model
output2 = model.predict(A)
print("output2.shape=", output2.shape)
##print("output2=", output2)
