'''
ref1: https://arxiv.org/pdf/1409.4842.pdf
ref2: https://becominghuman.ai/understanding-and-coding-inception-module-in-keras-eb56e9056b4b
'''
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers   import Input, Conv2D, Dense
from tensorflow.keras.layers   import Flatten, MaxPooling2D, GlobalAveragePooling2D
import numpy as np
import matplotlib.pyplot as plt

#1:
##import tensorflow as tf
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)

#2:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') # (50000, 32, 32, 3)
x_test  = x_test.astype('float32')  # (10000, 32, 32, 3)
x_train = x_train / 255.0
x_test  = x_test / 255.0

# one-hot encoding 
y_train = tf.keras.utils.to_categorical(y_train)
y_test  = tf.keras.utils.to_categorical(y_test)

#3: simple Inception_layer
inputs = Input(shape=(32, 32, 3))
L1 = Conv2D(64, (1,1), padding='same', activation='relu', name="L1")(inputs)
L2 = Conv2D(64, (3,3), padding='same', activation='relu', name="L2")(L1)

L3 = Conv2D(64, (1,1), padding='same', activation='relu')(inputs)
L3 = Conv2D(64, (5,5), padding='same', activation='relu', name="L3")(L3)

L4 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)
L4 = Conv2D(64, (1,1), padding='same', activation='relu', name="L4")(L4)
output = tf.keras.layers.concatenate([L1, L2, L3, L4], axis = 3)

#4: create top for cifar10 classification
output = Flatten()(output)
outs   = Dense(10, activation='softmax')(output)
model  = tf.keras.Model(inputs=inputs, outputs=outs)
model.summary()

#5: train and evaluate the model                 
opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
ret = model.fit(x_train, y_train, epochs=30, batch_size=128, 
                validation_split=0.2, verbose=0)
 
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

#6: plot accuracy and loss
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].plot(ret.history['loss'],  "g-")
ax[0].set_title("train loss")
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

ax[1].set_ylim(0, 1.1)
ax[1].plot(ret.history['accuracy'],     "b-", label="train accuracy")
ax[1].plot(ret.history['val_accuracy'], "r-", label="val accuracy")
ax[1].set_title("accuracy")
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')
plt.legend(loc='lower right')
fig.tight_layout()
plt.show()
