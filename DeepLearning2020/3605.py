import tensorflow as tf
import numpy as np
 
#1: crate a 1D input data with 3-channels
A = np.array([1, 2, 3, 4, 5]).astype('float32')
A = np.reshape(A, (1, -1, 1)) # (batch, steps, channels)
 
#2: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=A.shape[1:]))  # shape=(5, 1)
model.add(tf.keras.layers.ZeroPadding1D(padding=(1, 2)))
model.summary()

#3: apply A to model
output = model(A)
print("type(output) =", type(output))
print("output.numpy()=", output.numpy())

#4: apply A to model
output2 = model.predict(A)
print("type(output2) =", type(output2))
print("output2=", output2)
