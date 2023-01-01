import tensorflow as tf
from tensorflow.keras.preprocessing import image # pip install pillow
import numpy as np
import matplotlib.pyplot as plt

#1: load image
img_path = "./data/dog.jpg"
img = image.load_img(img_path,
                     target_size=(224, 224)) #(img_height, img_width)
img = image.img_to_array(img) # (224, 224, 3)
 
#2: transform img: 3D tensor
outs = []
labels = []
outs.append(img) # original image
labels.append("original")

outs.append(image.apply_affine_transform(img, theta= 30))
labels.append("theta= 30")

outs.append(image.apply_affine_transform(img, theta= 60))
labels.append("theta= 60")

outs.append(image.apply_affine_transform(img, theta= 90))
labels.append("theta= 90")

outs.append(image.apply_affine_transform(img, tx= 0, ty=50))
labels.append("tx= 0, ty=50")

outs.append(image.apply_affine_transform(img, shear= 50))
labels.append("shear= 50")

outs.append(image.apply_affine_transform(img, zx= 0.5, zy=1.0)) # zoom
labels.append("zx= 0.5, zy=1.0")

outs.append(image.apply_brightness_shift(img, brightness= 0.5))
labels.append("brightness= 0.5")

#3: save images in outs
img_path = "./data/transformed/"
for i in range(8):
    img=image.array_to_img(outs[i])
    image.save_img(img_path+str(i)+".png", img)
 
#4: display images in outs 
fig = plt.figure(figsize=(8, 4))
for i in range(8):   
    plt.subplot(2, 4, i + 1)  
    plt.imshow(outs[i].astype('uint8'))
    plt.title(labels[i])
    plt.axis("off")
    
fig.tight_layout()
plt.show() 
