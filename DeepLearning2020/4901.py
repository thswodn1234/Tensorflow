import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image # pip install pillow
import numpy as np
import matplotlib.pyplot as plt

#1: load image
img_path = "./data/dog.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
##img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
img = tf.expand_dims(img, axis=0)   # (1, 224, 224, 3)

#2:random image augmentation
# ref: https://keras.io/ko/preprocessing/image/
##datagen = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,    
##                             height_shift_range=0.1, zoom_range=0.2)
##datagen = ImageDataGenerator(width_shift_range=0.4)  #[-10, 10]
##datagen = ImageDataGenerator(height_shift_range=0.2)
##datagen = ImageDataGenerator(horizontal_flip=True)
##datagen = ImageDataGenerator(vertical_flip =True)
##datagen = ImageDataGenerator(brightness_range= [0.2, 1.0])
##datagen = ImageDataGenerator(zoom_range=0.4) #[0.6, 1.4])
datagen = ImageDataGenerator(rotation_range=90)

it = datagen.flow(img, batch_size=1)

#3: generate and display    
fig = plt.figure(figsize=(8, 4))
for i in range(8):
    if i == 0:
        batch = img # original image
    else:
        batch = it.next() # generate an image from datagen
    plt.subplot(2, 4, i + 1)  
    plt.imshow(tf.cast(batch[0], tf.uint8)) ## plt.imshow(batch[0].astype('uint8'))        
    plt.axis("off")
fig.tight_layout()
plt.show() 
