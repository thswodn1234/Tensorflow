#ref: https://www.tensorflow.org/tutorials/images/classification
import tensorflow as tf
from tensorflow.keras.layers   import Input, Conv2D, MaxPool2D, Dense  
from tensorflow.keras.layers   import BatchNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


#2: build a model with functional API
def create_cnn2d(input_shape=(224, 224, 3), num_class = 2):
    inputs = Input(shape=input_shape)
    x=Conv2D(filters=16, kernel_size = (3,3), activation='relu')(inputs)
    x=BatchNormalization()(x)
    x=MaxPool2D()(x)

    x=Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x)
    x=MaxPool2D()(x)
    x=Dropout(rate=0.5)(x)
      
    x=Flatten()(x)
    x= Dense(units=256, activation='relu')(x)
    x= Dropout(rate=0.5)(x)
    outputs= Dense(units=num_class, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    
    opt = RMSprop(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_cnn2d()
##model.summary()

#3: image augmentation
#3-1:
train_datagen = ImageDataGenerator(
    rescale = 1./255, 
    rotation_range=20,    
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2)

test_datagen = ImageDataGenerator(rescale= 1./255)

#3-2:
img_width, img_height = 224, 224
train_dir= "C:/Users/user/.keras/datasets/cats_and_dogs_filtered/train"
test_dir = "C:/Users/user/.keras/datasets/cats_and_dogs_filtered/validation"
train_generator= train_datagen.flow_from_directory(
    train_dir, target_size=(img_width, img_height), batch_size=32,
    class_mode="categorical", subset='training')
valid_generator= train_datagen.flow_from_directory(
    train_dir, target_size=(img_width, img_height), batch_size=32,
    class_mode="categorical", subset='validation')

test_generator= test_datagen.flow_from_directory(
    test_dir, target_size=(img_width, img_height), batch_size=32,
    class_mode="categorical")

print("train_generator.class_indices=", train_generator.class_indices)
print("test_generator.class_indices=", test_generator.class_indices)

print("train_generator.classes.shape=", train_generator.classes.shape)
print("valid_generator.classes.shape=", valid_generator.classes.shape)
print("test_generator.classes.shape=",  test_generator.classes.shape)

train_steps= int(np.ceil(train_generator.classes.shape[0]/train_generator.batch_size))
valid_steps= int(np.ceil(valid_generator.classes.shape[0]/valid_generator.batch_size))
test_steps= int(np.ceil(test_generator.classes.shape[0]/test_generator.batch_size))
print("train_steps=",train_steps)
print("valid_steps=",valid_steps)
print("test_steps=",test_steps)

#4: train the model using generator
ret = model.fit(train_generator, epochs=100,  
                validation_data=valid_generator,
                steps_per_epoch= train_steps,
                validation_steps=valid_steps,
                verbose=2)

#5:  
#5-1: calculate confusion_matrix(C)
y_pred = model.predict(train_generator, steps=train_steps, verbose=2)
y_label = np.argmax(y_pred, axis = 1)
C = tf.math.confusion_matrix(train_generator.labels, y_label)
print("confusion_matrix(C):", C)

#5-2: evaluate
train_loss, train_acc = model.evaluate(train_generator,
                                       steps = train_steps,
                                       verbose=2)
test_loss, test_acc = model.evaluate(test_generator,
                                     steps = test_steps,
                                     verbose=2)

#6: plot accuracy and loss
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
