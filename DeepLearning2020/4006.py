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

#3: build a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(x_train.shape[1:])) # shape=(32, 32, 3)
##model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.AveragePooling2D())
model.summary()

#4: apply x_train to model
output = model.predict(x_train[:8]) #(batch, pooled_rows, pooled_cols, channels)
img = output/255   # output.astype('uint8')
print("img.shape=", img.shape)

#5: display images
fig = plt.figure(figsize=(8, 4))
for i in range(8):   
    plt.subplot(2, 4, i + 1)  
    plt.imshow(img[i])
    plt.axis("off")
fig.tight_layout()
plt.show()
