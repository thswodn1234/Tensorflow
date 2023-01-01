import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

# expand data with channel = 1
x_train = np.expand_dims(x_train,axis=3) # (60000, 28, 28, 1)
x_test  = np.expand_dims(x_test, axis=3) # (10000, 28, 28, 1)

#3: Sobel kernel initial values, shape: (2, 3, 3)
W = np.array([[[-1, 0, 1],      
               [-2, 0, 2],
               [-1, 0, 1]],
              [[-1,-2,-1],      
               [ 0, 0, 0],
               [ 1, 2, 1]]], dtype='float32')
W = np.transpose(W, (1, 2, 0)) # (3, 3, 2)
W=np.expand_dims(W, axis=2) #(3,3,1,2)=(kernel_size[0],kernel_size[1],channels, filters)

#4: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(x_train.shape[1:])) # shape=(28, 28, 1)
model.add(tf.keras.layers.Conv2D(filters=2,
                                 kernel_size = W.shape[:2], # (3, 3)
                                 use_bias=False,
                                 kernel_initializer=tf.constant_initializer(W)))
model.summary()

#5: apply x_train to model
##output = model.predict(x_train[:8])
output = model(x_train[:8]) # (batch, new_rows, new_cols, filters)
mag = tf.sqrt(tf.square(output[:,:,:,0])+tf.square(output[:,:,:,1]))
max_mag = tf.reduce_max(mag)  # tf.norm(mag, np.inf)
mag = tf.divide(mag, max_mag) # range[ 0, 1]
img = mag.numpy()
print("img.shape=", img.shape)

#6: display images
fig = plt.figure(figsize=(8, 4))
for i in range(8):   
    plt.subplot(2, 4, i + 1)  
    plt.imshow(img[i], cmap='gray')
    plt.axis("off")
fig.tight_layout()
plt.show()

#6: weights
W2 = model.trainable_variables[0] # (kernel_size[0], kernel_size[1], channels, filters)
print("W2.shape=", W2.shape)
print("W2[:,:,0,0]=", W2[:,:,0,0]) # 0-channel, 0-filter
print("W2[:,:,0,1]=", W2[:,:,0,1]) # 0-channel, 1-filter
