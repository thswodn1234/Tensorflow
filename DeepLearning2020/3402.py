import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
#1
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#2: normalize images
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255.0 # [0, 1]
x_test  /= 255.0
##x_train -= 0.5
##x_test -= 0.5

#3: preprocessing the target(y_train, y_test)
y_train = y_train.flatten() 
y_test  = y_test.flatten()

# one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train) # (50000, 10)
y_test = tf.keras.utils.to_categorical(y_test)   # (10000, 10)

#4: build a model with weight regularization
init = 'he_uniform'
act = tf.keras.layers.LeakyReLU(alpha=0.3) # 'relu'
reg = tf.keras.regularizers.l2(0.001)       # 0.01
n = 100

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer = init,
                                 kernel_regularizer=reg))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer = init,
                                 kernel_regularizer=reg))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
##model.summary()
 
#4-1: configure the model for training
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)  # 'rmsprop'
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
                                 
#4-2: train and evaluate the model
ret2 = model.fit(x_train, y_train, epochs=201, batch_size=400,
                                    validation_data = (x_test, y_test), verbose=0)
train_loss2, train_acc2 = model.evaluate(x_train, y_train, verbose=2)
test_loss2,  test_acc2  = model.evaluate(x_test,  y_test,  verbose=2)

#4-3: plot accuracy
plt.plot(ret2.history['accuracy'],     "b-", label="train accuracy")
plt.plot(ret2.history['val_accuracy'], "r-", label="val accuracy")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc="best")
plt.show()
