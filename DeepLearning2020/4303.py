import tensorflow as tf
import numpy as np

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: 1D input data: A, B
A = np.array([1, 2, 3, 4, 5]).astype('float32')
B = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype('float32')
A = np.reshape(A, (1, -1, 1)) # (batch, steps, channels)
B = np.reshape(B, (1, -1, 1)) # (batch, steps, channels) 

#3: build a model
input_x = tf.keras.layers.Input(shape=A.shape[1:])
input_y = tf.keras.layers.Input(shape=B.shape[1:])
 
x = tf.keras.layers.MaxPool1D()(input_x)
y = tf.keras.layers.MaxPool1D()(input_y) 

pad = y.shape[1] - x.shape[1] # 2
x = tf.keras.layers.ZeroPadding1D(padding=(0, pad))(x)

out2 = tf.keras.layers.Add()([x, y])
##out2 = tf.keras.layers.Subtract()([x, y])
##out2 = tf.keras.layers.Multiply()([x, y])
##out2 = tf.keras.layers.Minimum()([x, y])
##out2 = tf.keras.layers.Maximum()([x, y])
##out2 = tf.keras.layers.Average()([x, y])
out3 = tf.keras.layers.Concatenate()([x, y])
out4 = tf.keras.layers.Dot(axes=[1,1])([x, y]) # inner product
out5 = tf.keras.layers.Dot(axes=-1)([x, y])    # outer product 

out_list = [x, y, out2, out3, out4, out5]
model = tf.keras.Model(inputs=[input_x, input_y], outputs= out_list)
##model.summary()
print("model.output_shape=", model.output_shape)

#4: apply [A, B] to model
##output = model([A, B])      # Tensor output
output = model.predict([A, B]) # numpy output
for i in range(len(output)):
    print("output[{}]={}".format(i, output[i]))
