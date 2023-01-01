import tensorflow as tf
from tensorflow.keras.preprocessing import image # pip install pillow
import numpy as np
import matplotlib.pyplot as plt

#1: load image
img_path = "./data/dog.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)  # (224, 224, 3)
 
#2:random transform img: 3D
outs = []
outs.append(img) # original image
#2-1
##for i in range(7):
##    outs.append(image.random_brightness(img, brightness_range=[0.2, 1.0]))

#2-2  
##for i in range(7):
##    outs.append(image.random_shift(img, wrg= 0.4, hrg= 0.0,
##                      row_axis=0, col_axis=1, channel_axis=2))

#2-3
##for i in range(7):
##    outs.append(image.random_shear(img, intensity=40, # intensity in degrees
##                      row_axis=0, col_axis=1, channel_axis=2))

#2-4
##for i in range(7):
##    outs.append(image.random_rotation(img, rg=20,
##                      row_axis=0, col_axis=1, channel_axis=2))

#2-5
for i in range(7):    
    outs.append(image.random_zoom(img, zoom_range= [0.4, 1.6],
                      row_axis=0, col_axis=1, channel_axis=2))
     
#3: display  
fig = plt.figure(figsize=(8, 4))
for i in range(8):   
    plt.subplot(2, 4, i + 1)  
    plt.imshow(outs[i].astype('uint8'))
    plt.axis("off")
fig.tight_layout()
plt.show() 
