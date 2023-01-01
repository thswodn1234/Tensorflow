import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
##print("x_train.shape=", x_train.shape) # (50000, 32, 32, 3)
##print("x_test.shape=",  x_test.shape)  # (10000, 32, 32, 3)

W = np.array([[[-1, 0, 1],      
               [-2, 0, 2],
               [-1, 0, 1]],
              [[-1,-2,-1],      
               [ 0, 0, 0],
               [ 1, 2, 1]]], dtype='float32')


#3: convert W.shape=(2, 3, 3) to (kernel_size[0], kernel_size[1], channels, filters)
W = np.transpose(W, (1, 2, 0)) # (3, 3, 2)
W = np.expand_dims(W, axis=2)  # (3, 3, 1, 2)
W = np.concatenate((W, W, W), axis= 2) # (3, 3, 3, 2) # channels=3, filters = 2

#4: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(x_train.shape[1:])) # shape=(32, 32, 3)
model.add(tf.keras.layers.Conv2D(filters=2,
                                 kernel_size = W.shape[:2], # (3, 3)
##                                 dilation_rate=(2,2),
##                                 padding='same',
                                 use_bias=False,
                                 kernel_initializer=tf.constant_initializer(W))) 
model.summary()

#5: apply x_train to model
##output = model.predict(x_train[:8])
output = model(x_train[:8]) # (batch, new_rows, new_cols, filters)
gx = output[:,:,:,0]
gy = output[:,:,:,1]
mag = tf.sqrt(tf.square(gx)+tf.square(gy))
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

#7: weights
W2 = model.trainable_variables[0] # (kernel_size[0], kernel_size[1], channels, filters)
print("W2.shape=", W2.shape)
print("0-filter: gx")
print("W2[:,:,0,0]=", W2[:,:,0,0]) # 0-channel, 0-filter
print("W2[:,:,1,0]=", W2[:,:,1,0]) # 1-channel, 0-filter
print("W2[:,:,2,0]=", W2[:,:,2,0]) # 2-channel, 0-filter

print("1-filter: gy")
print("W2[:,:,0,1]=", W2[:,:,0,1]) # 0-channel, 1-filter
print("W2[:,:,1,1]=", W2[:,:,1,1]) # 1-channel, 1-filter
print("W2[:,:,2,1]=", W2[:,:,2,1]) # 2-channel, 1-filter
