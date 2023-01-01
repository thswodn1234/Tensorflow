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
x_train = np.expand_dims(x_train,axis = 3)      # (60000, 28, 28, 1)
x_test  = np.expand_dims(x_test, axis = 3)      # (10000, 28, 28, 1)

#3: kernel
W = np.array([[ 1,  1],      
              [ 1,  1]], dtype = 'float32')
W = W.reshape(2, 2, 1, 1)   # (kernel_size[0], kernel_size[1], filters, channels)

#4: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(x_train.shape[1:])) # shape = (28, 28, 1)
model.add(tf.keras.layers.Conv2DTranspose(filters=1,
                                 kernel_size = (2, 2),
                                 strides = (2, 2),
                                 padding = 'valid',
                                 use_bias = False,
                                 kernel_initializer = tf.constant_initializer(W)))
model.summary()

#5: apply x_train to model
output = model.predict(x_train[:8])  # (8, 56, 56, 1)
img = output[:,:,:,0]                # 0-filters                  
print("img.shape=", img.shape)

#6: display images
fig = plt.figure(figsize = (8, 4))
for i in range(8):   
    plt.subplot(2, 4, i + 1)  
    plt.imshow(img[i], cmap = 'gray')
    plt.axis("off")
fig.tight_layout()
plt.show()
