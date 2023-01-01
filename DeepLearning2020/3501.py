import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
#1
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# subsampling for overfitting
n_sample = 6000
x_train = x_train[:n_sample]
y_train = y_train[:n_sample]

#2:normalize images
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255.0 # [0, 1]
x_test  /= 255.0

#3: one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train) # (n_sample, 10)
y_test = tf.keras.utils.to_categorical(y_test)   # (10000,    10)

#4: build a model with dropout
act = "relu"
init = "he_uniform"

n = 100
dropout_rate = 0.2  # 0.5
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer = init))
model.add(tf.keras.layers.Dropout( rate=dropout_rate))

model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer = init))
model.add(tf.keras.layers.Dropout( rate=dropout_rate))

model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.summary()

#4-1: configure the model for training
##opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#4-2: train and evaluate the model
ret = model.fit(x_train, y_train, epochs=201, batch_size=400,
                                  validation_data = (x_test, y_test), verbose=0)
 
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc   = model.evaluate(x_test,  y_test, verbose=2)

#4-3: plot accuracies
plt.title("Dropout rate = %s, %s traing data in mnist"%(dropout_rate, n_sample))
plt.plot(ret.history['accuracy'],     "b-", label="train accuracy")
plt.plot(ret.history['val_accuracy'], "r-", label="val accuracy")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc="best")
plt.show()
