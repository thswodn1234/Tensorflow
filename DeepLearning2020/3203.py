import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import datetime
#1
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#2: normalize images
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255.0 # [0, 1]
x_test  /= 255.0

#3: one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train) # (60000, 10)
y_test = tf.keras.utils.to_categorical(y_test)   # (10000, 10)

#4: build a model
#4-1
##init = tf.keras.initializers.he_normal() # 'he_normal'
##act =  tf.keras.activations.relu         # 'relu'

#4-2
##init = tf.keras.initializers.he_normal()   # 'he_normal'
##act = tf.keras.layers.LeakyReLU(alpha=0.3)

#4-3
init =  tf.keras.initializers.he_uniform() # 'he_uniform'
act = tf.keras.layers.LeakyReLU(alpha=0.3)

n = 100
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=n, activation=act, kernel_initializer=init))
model.add(tf.keras.layers.Dense(units=10,activation='softmax', kernel_initializer=init))
model.summary()
opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#5: creates a summary file writer for the given log directory
import os
path = "c:\\tmp\\logs\\"
if not os.path.isdir(path):
    os.mkdir(path)
##logdir = path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = path + "3203"

file_writer = tf.summary.create_file_writer(logdir + "/gradient")
file_writer.set_as_default()

#6:  calculate averages and histograms of gradients in layers
class GradientCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, freq=10):
##        super(GradientCallback, self).__init__()
        self.freq = freq

    def on_epoch_end(self, epoch, logs):
        if epoch%self.freq != 0:
            return
        with tf.GradientTape() as tape:
            y_pred = model(x_train)  # tensor, logits
            loss   = tf.keras.losses.binary_crossentropy(y_train, y_pred)
        grads = tape.gradient(loss, model.trainable_weights)
        for n in range(1, len(model.layers)):
            i2 = (n-1)*2 # weights
            i1 = i2 + 1  # biases

            bias_avg   = tf.reduce_mean(tf.abs(grads[i1]))
            weight_avg = tf.reduce_mean(tf.abs(grads[i2]))
            
            tf.summary.scalar("layer_%d/avg/bias"%n, data=bias_avg, step=epoch)   
            tf.summary.scalar("layer_%d/avg/weight"%n, data=weight_avg, step=epoch)
##            
            tf.summary.histogram("layer_%d/hist/bias"%n, data=grads[i1], step=epoch)
            tf.summary.histogram("layer_%d/hist/weight"%n, data=grads[i2], step=epoch)
            
    def on_train_end(self, logs):
        tf.summary.flush()

callback1 = GradientCallback() # freq = 10
callback2 = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq= 10) #profile_batch=0      
                                  
#7: train and evaluate the model
ret = model.fit(x_train, y_train, epochs=101, batch_size=200, validation_split=0.2,
                 verbose=2, callbacks=[callback1, callback2])

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
