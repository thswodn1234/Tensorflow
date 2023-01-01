import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers   import Input, Conv2D, MaxPool2D, Dense  
from tensorflow.keras.layers   import BatchNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2
(x_train, y_train), (x_test, y_test) = cifar100.load_data() # 'fine'
x_train = x_train.astype('float32') # (50000, 32, 32, 3)
x_test  = x_test.astype('float32')  # (10000, 32, 32, 3)

# one-hot encoding 
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#3: build a model with functional API
def create_cnn2d(input_shape, num_class = 100):
    inputs = Input(shape=input_shape) #  shape=(32, 32, 3)
    x= Conv2D(filters=16, kernel_size = (3,3), padding='same',
              activation='relu', )(inputs)
    x= BatchNormalization()(x)
    x= MaxPool2D()(x)
    
    x= Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(x)
    x= BatchNormalization()(x)
    x= MaxPool2D()(x)
    x= Dropout(rate=0.25)(x)

    x= Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(x)
    x= BatchNormalization()(x)
    x= MaxPool2D()(x)
    x= Dropout(rate=0.5)(x)
      
    x=Flatten()(x)
    
    x = Dense(units=256, activation='relu')(x)
    x= Dropout(rate=0.2)(x)
    outputs = Dense(units=num_class, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    opt = RMSprop(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])    
    return model
model = create_cnn2d(input_shape = x_train.shape[1:])
##model.summary()

#4: image augmentation
datagen = ImageDataGenerator( # ref: https://keras.io/ko/preprocessing/image/
    featurewise_center = True,            # mean = 0.0
    featurewise_std_normalization= True,  # std = 1.0
    rotation_range=20,    
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    validation_split=0.2)

datagen.fit(x_train) #  computes the internal data stats: mean, std, zca
##print("datagen.mean = ", datagen.mean)
##print("datagen.mean = ",  datagen.std)

train_generator= datagen.flow(x=x_train, y=y_train, batch_size=400, subset='training')
valid_generator= datagen.flow(x=x_train, y=y_train, batch_size=400, subset='validation')

train_steps= int(np.ceil(train_generator.n/train_generator.batch_size))
valid_steps= int(np.ceil(valid_generator.n/valid_generator.batch_size))
print("train_steps=", train_steps)
print("valid_steps=", valid_steps)

#5: train the model using generator
ret = model.fit(train_generator, epochs=100,
                validation_data=valid_generator,
                steps_per_epoch= train_steps,
                validation_steps=valid_steps,
                verbose=0)

#6:  predict and evaluate the model
#6-1: normalize x_train, x_test, the same as datagen
datagen.standardize(x_train) # mean=0, std=1
datagen.standardize(x_test)  # mean=0, std=1

#6-2: calculate confusion_matrix(C)
##y_pred = model.predict(x_train)
##y_label = np.argmax(y_pred, axis = 1)
##C = tf.math.confusion_matrix(np.argmax(y_train, axis = 1), y_label)
##print("confusion_matrix(C):", C)

#6-3: evaluate
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

#7: plot accuracy and loss
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].plot(ret.history['loss'],  "g-")
ax[0].set_title("train loss")
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

ax[1].plot(ret.history['accuracy'],     "b-", label="train accuracy")
ax[1].plot(ret.history['val_accuracy'], "r-", label="val_accuracy")
ax[1].set_title("accuracy")
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')
plt.legend(loc="best")
fig.tight_layout()
plt.show()
