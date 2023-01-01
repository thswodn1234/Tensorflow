import tensorflow as tf
import numpy as np

#1
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: crate a 1D input data
A = np.array([[1, 2, 3, 4, 5],
              [1, 1, 1, 1, 1],
              [1, 2, 0, 1, 2]], dtype='float32')
n = 2 # number of neurons in Dense, # of filters in Conv1D
steps = A.shape[1] # length, 5

#3: kernel initial values, shape: (5, 2)
W = np.array([[1., 2.],      
              [1., 2.],
              [1., 2.],
              [1., 2.],
              [1., 2.]], dtype="float")

#4: Dense with n units, input_dim = steps
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=steps ))
model.add(tf.keras.layers.Dense(units=n, use_bias=False , # input_dim=steps, 
                                 kernel_initializer=tf.constant_initializer(W)))
model.summary()
print("model.trainable_variables=", model.trainable_variables)

# apply A to model 
##output = model(A)      # tensor, output.shape= (3, 2)
output = model.predict(A) # numpy, output.shape= (3, 2)
print("output=", output)

#5: Conv1D with n filters, kernel_size =steps, strides = 1, input shape=(steps,1)
model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Input(shape=(steps,1)))
model2.add(tf.keras.layers.Conv1D(filters = n, kernel_size = steps, use_bias=False,                                        kernel_initializer= tf.constant_initializer(W))) 
model2.add(tf.keras.layers.Flatten()) # output.shape : (batch, new_steps*filters)
model2.summary()
print("model2.trainable_variables=", model2.trainable_variables)

# apply A to model2
A2 = np.expand_dims(A, axis=2) # tf.expand_dims(A, axis=2), shape = ([3, 5, 1])
##print("A2 = ", A2)
output2 = model2.predict(A2) # output2.shape= (3, 2)
print("output2=", output2)
