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

opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#5
##ref1: https://www.tensorflow.org/guide/keras/custom_callback
##ref2: https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e
class PlotLoss(tf.keras.callbacks.Callback):
    
    def __init__(self, epoches, close=False):
        self.nepoches = epoches
        self.close = close
        
    def on_train_begin(self, logs): 
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        plt.ion() # interactive on
        self.fig = plt.figure(figsize=(8, 6))
        self.ax =  plt.gca()

        self.line1, = self.ax.plot([], [], "b-", lw= 2, label="loss")
        self.line2, = self.ax.plot([], [], "r-", lw= 2, label="val_loss")

        self.ax.set_xlim(0, self.nepoches)
        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel("loss")
        self.ax.legend(loc="upper right")
        plt.show(); plt.pause(0.01)
##        self.logs = []
        
    def on_train_end(self, logs): 
        if self.close:
            plt.close(self.fig) #  plt.close("all")       
        plt.ioff()
        
    def on_epoch_end(self, epoch, logs):# logs: dict
##        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        self.ax.set_title("epoch : {}".format(epoch))

        self.line1.set_data(self.x, self.losses)
        self.line2.set_data(self.x, self.val_losses)
        
        self.ax.relim() # recompute the data limits
        # autoscale the view limits using the data limit
        self.ax.autoscale_view(tight=True,scalex=False,scaley=True)      
        plt.pause(0.01)
        
n_epoches = 100
callback = PlotLoss(n_epoches) # create callback instance

#6
ret = model.fit(x_train, y_train, epochs=n_epoches, batch_size=200,
                validation_split=0.2, verbose=2, callbacks=[callback])
