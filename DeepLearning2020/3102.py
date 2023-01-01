import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import datetime
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
model.summary()

opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#5
import os
path = "c:\\tmp\\logs\\"
if not os.path.isdir(path):
    os.mkdir(path)
##logdir = path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = path + "3102"

file_writer = tf.summary.create_file_writer(logdir + "/custom")
file_writer.set_as_default()

#6, ref: https://www.tensorflow.org/tensorboard/scalars_and_keras
def scheduler(epoch, lr): # ref: [step29_02]
    if epoch % 10 == 0 and epoch:        
        lr = 0.1*lr
    tf.summary.scalar("learning rate", data=lr, step=epoch)   
    return lr

#7: get a model from layer n
def getLayerModel(n):
    layer_model = tf.keras.models.Model(
                                 inputs=model.input,
                                 outputs=model.layers[n].output)
    return layer_model

#8
class OutputCallback(tf.keras.callbacks.Callback): # ref: [step29_03]  
    def on_epoch_end(self, epoch, logs):
        for n in range(1, len(model.layers)):
            layer_model = getLayerModel(n)
##            output = layer_model.predict(x_train) # numpy
            output   = layer_model(x_train)         # tensor
            tf.summary.histogram("layer_%d"%n, data=output, step=epoch)

    def on_train_end(self, logs):
        tf.summary.flush()

callback1 = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
callback2 = OutputCallback()
callback3 = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq= 2)       
                                  
#9
ret = model.fit(x_train, y_train, epochs=100, batch_size=200, validation_split=0.2,
                 verbose=2, callbacks=[callback1, callback2, callback3])
