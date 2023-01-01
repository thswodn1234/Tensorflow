# !pip install –U tfds-nightly # in Colab
import tensorflow as tf
import tensorflow_datasets as tfds # pip install tensorflow_datasets
tfds.disable_progress_bar()
 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate 
from tensorflow.keras.layers import Dense,  Flatten, Reshape, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, UpSampling2D

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image # pip install pillow

import numpy as np
import matplotlib.pyplot as plt

#1: ref [step37_01], [그림 2.9]
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


#2: [step55_04]
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

#2-1
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
  input_image, input_mask = normalize(input_image, input_mask)
##  return input_image, input_mask
  species = datapoint['species'] 
  return input_image, input_mask, species

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
  input_image, input_mask = normalize(input_image, input_mask)
##  return input_image, input_mask  
  species = datapoint['species']    
  return input_image, input_mask, species

#2-2 
BATCH_SIZE = 16
TRAIN_STEPS = info.splits['train'].num_examples // BATCH_SIZE
BUFFER_SIZE = 1000

train_ds = dataset['train'].map(load_image_train)
train_ds = train_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
##train_ds = train_ds.batch(BATCH_SIZE)

test_ds  = dataset['test'].map(load_image_train)
test_ds  = test_ds.batch(BATCH_SIZE)

#3:[step57_02]
def unet_2(input_shape=(128, 128, 3), num_classes=3):
  
    inputs = Input(shape=input_shape)
    # 128
    
    down1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Conv2D(32, (3, 3), activation='relu', padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1_pool = MaxPool2D()(down1)
    # 64
    
    down2 = Conv2D(64, (3, 3), activation='relu', padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Conv2D(64, (3, 3), activation='relu', padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2_pool = MaxPool2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(128, (3, 3), activation='relu', padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Conv2D(128, (3, 3), activation='relu', padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3_pool = MaxPool2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(256, (3, 3), activation='relu', padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Conv2D(256, (3, 3), activation='relu', padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4_pool = MaxPool2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(512, (3, 3), activation='relu', padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Conv2D(512, (3, 3), activation='relu', padding='same')(center)
    center = BatchNormalization()(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    
    up4 = Conv2D(256, (3, 3), activation='relu', padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Conv2D(256, (3, 3), activation='relu', padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Conv2D(256, (3, 3), activation='relu', padding='same')(up4)
    up4 = BatchNormalization()(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(128, (3, 3), activation='relu', padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Conv2D(128, (3, 3), activation='relu', padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Conv2D(128, (3, 3), activation='relu', padding='same')(up3)
    up3 = BatchNormalization()(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)    
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    # 128

    classify = Conv2D(3, (1, 1), padding='same', activation='sigmoid')(up1)   
    model = Model(inputs=inputs, outputs=classify)
    
    model.compile(optimizer=RMSprop(0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])    
    return model

#4:
UNET = unet_2()
##UNET.summary()
ret = UNET.fit(train_ds, epochs = 30,
               steps_per_epoch=TRAIN_STEPS,  verbose = 0)

TEST_STEPS = info.splits['test'].num_examples // BATCH_SIZE
train_loss, train_acc = UNET.evaluate(train_ds, steps=TRAIN_STEPS, verbose=2)
test_loss, test_acc   = UNET.evaluate(test_ds, steps=TEST_STEPS, verbose=2)

#5  
#5-1:  
def create_mask(pred_mask):  # (:, 128, 128, 3)
  pred_mask = tf.argmax(pred_mask, axis=-1) # (:, 128, 128), axis=3 
  pred_mask = pred_mask[..., tf.newaxis]    # (:, 128, 128, 1) 
  return pred_mask

#5-2: display a batch
label_name = ['Cat', 'Dog']
def display_images(dataset):
    n = min([4, BATCH_SIZE]) # at most 4    
    fig = plt.figure(figsize=(n*2, 6))  # (8, 6)

    for images, masks,species in dataset.take(1): # 1 batch
      pred_mask = UNET.predict(images)
      pred_mask = create_mask(pred_mask)
      for i in range(n): # n of len(images)
        plt.subplot(3, n, i+1) # 0-row: images
        a_img = image.array_to_img(images[i])
        plt.imshow(a_img)  
        plt.title(label_name[species[i]])
        plt.axis("off")
        
        plt.subplot(3, n, i+1+n)   # 1-row: corrected mask
        a_img = image.array_to_img(masks[i])
        plt.imshow(a_img)
        plt.axis("off")

        plt.subplot(3, n, i+1+2*n) # 2-row: predicted mask
        a_img = image.array_to_img(pred_mask[i])
        plt.imshow(a_img)
        plt.axis("off")

    fig.tight_layout()
    plt.show()

display_images(test_ds)    
##display_images(train_ds)    




