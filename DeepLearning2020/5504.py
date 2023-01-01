'''
ref: https://www.tensorflow.org/tutorials/images/segmentation
'''
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds # pip install tensorflow_datasets
##tfds.disable_progress_bar()

#1:
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
##print("info=", info)
##
##print("info.features['label'].num_classes=", info.features['label'].num_classes) # 37
##print("info.features['label'].names=", info.features['label'].names)
##
##print("info.features['species'].num_classes=", info.features['species'].num_classes) # 2
##print("info.features['species'].names=", info.features['species'].names) #['Cat', 'Dog']
##
##print("info.splits['train'].num_examples=", info.splits['train'].num_examples)
##print("info.splits['test'].num_examples=",  info.splits['test'].num_examples)

#2:
ds = dataset['train'] #ds = dataset['test']
for i, example in enumerate(ds.take(2)):
  name, label, species = example["file_name"], example["label"], example["species"]
  image, mask = example["image"], example["segmentation_mask"]
  print("example[{}]: name:{}, label:{},species:{}, image.shape={}, mask.shape={}".format(
                              i, name.numpy(), label, species, image.shape, mask.shape))

#3: batch
#3-1  
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
  input_image, input_mask = normalize(input_image, input_mask)
  species = datapoint['species']
  
  return input_image, input_mask, species

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
  input_image, input_mask = normalize(input_image, input_mask)
  species = datapoint['species']
    
  return input_image, input_mask, species

#3-2 
BATCH_SIZE = 4
train_ds = dataset['train'].map(load_image_train)
test_ds  = dataset['test'].map(load_image_train)

train_ds = train_ds.batch(BATCH_SIZE)
test_ds  = test_ds.batch(BATCH_SIZE)

#4: display a batch
label_name = ['Cat', 'Dog']
def display_images(dataset):

    n = BATCH_SIZE     
    fig = plt.figure(figsize=(n*2, 4))  # (8, 4)

    for images, masks,species in dataset.take(1): # 1 batch
      for i in range(len(images)): # BATCH_SIZE
        plt.subplot(2, n, i+1)
##        print("i={}, images[i].shape={}, images.shape={}".format(i, images[i].shape, images.shape))
        plt.imshow(images[i])
        plt.title(label_name[species[i]])
        plt.axis("off")
        
        plt.subplot(2, n, i+1+n)
        plt.imshow(masks[i])
        plt.axis("off")

    fig.tight_layout()
    plt.show()
    
display_images(train_ds)    
##display_images(test_ds)



