import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
#1
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#2:normalize images
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255.0 # [0, 1]
x_test  /= 255.0

#3: one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train) # (60000, 10)
y_test = tf.keras.utils.to_categorical(y_test)   # (10000, 10)

#4: x_train.shape = (60000, 28, 28)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=5, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
##model.summary()

opt = tf.keras.optimizers.RMSprop(learning_rate=1.0)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#5
##def scheduler(epoch):
##    lr = model.optimizer.lr.numpy() # tf.keras.backend.get_value(model.optimizer.lr)
##    if epoch % 2 == 0 and epoch:        
##        return 0.1*lr
##    return lr
def scheduler(epoch, lr):
    if epoch % 2 == 0 and epoch:
        return 0.1*lr
    return lr
callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

#6
ret = model.fit(x_train, y_train, epochs=10, batch_size=200,
                validation_split=0.2, verbose=0, callbacks=[callback])
