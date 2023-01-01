import tensorflow as tf
from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt

#1
(x_train, y_train), (x_test, y_test) = cifar100.load_data() #' fine'
##print("x_train.shape=", x_train.shape) # (50000, 32, 32, 3)
##print("y_train.shape=", y_train.shape) # (50000, 1)
##print("x_test.shape=",  x_test.shape)  # (10000, 32, 32, 3)
##print("y_test.shape=",  y_test.shape)  # (10000, 1)

#2:normalize images
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
def normalize_image(image): # 3-channel
    mean=  np.mean(image, axis = (0, 1, 2))
    std =  np.std(image,  axis = (0, 1, 2))
    image = (image-mean)/std
    return image
x_train= normalize_image(x_train) # range: N(mean=0, std=1]
x_test = normalize_image(x_test)

#3
nlabel, count = np.unique(y_train, return_counts=True)
nClass = len(nlabel) # 'fine': 100, 'coarse':20

#4: preprocessing the target(y_train, y_test)
y_train = y_train.flatten() 
y_test  = y_test.flatten()
##print("y_train.shape=", y_train.shape) # (50000,)
##print("y_test.shape=",  y_test.shape)  # (10000,)

# one-hot encoding: 'mse', 'categorical_crossentropy'
y_train = tf.keras.utils.to_categorical(y_train) # (50000, nClass)
y_test = tf.keras.utils.to_categorical(y_test)   # (10000, nClass)

#5
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Dense(units=100, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=nClass, activation='softmax'))
model.summary()

opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])
ret = model.fit(x_train, y_train, epochs=200, batch_size=200, 
               validation_split=0.2, verbose=2)

#6
y_pred = model.predict(x_train)
y_label = np.argmax(y_pred, axis = 1)
C = tf.math.confusion_matrix(np.argmax(y_train, axis = 1), y_label)
print("confusion_matrix(C):", C)
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

#7: plot accuracy and loss
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].plot(ret.history['loss'],  "g-")
ax[0].set_title("train loss")
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

ax[1].plot(ret.history['accuracy'],     "b-", label="train accuracy")
ax[1].plot(ret.history['val_accuracy'], "r-", label="val accuracy")
ax[1].set_title("accuracy")
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')
plt.legend(loc="best")
fig.tight_layout()
plt.show()
