import tensorflow as tf
from tensorflow.keras.layers     import Input, Lambda
from tensorflow.keras.preprocessing import image # pip install pillow
import numpy as np
import matplotlib.pyplot as plt

#1: 
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)

#2: input an image
img_path = './data/dog.jpg'    # './data/elephant.jpg'
img = image.load_img(img_path) #, target_size=(224, 224))
X = image.img_to_array(img)
X = np.expand_dims(X, axis=0)  # (1, img.height,img.width, 3)

#3: resize_layer
inputs = Input(shape=X.shape[1:])
resize_layer = Lambda(lambda x: tf.image.resize(x,(224, 224)))(inputs)
model  = tf.keras.Model(inputs = inputs, outputs= resize_layer)
model.summary()

#4: predict an image
output = model.predict(X)

#5: display 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5) )
size = X.shape[1:] 
max_height= max(size[0], output[0].shape[0])
max_width = size[1]+ output[0].shape[1]+0.1 # 0.1: space

# X[0] display
bottom, height = 0, X[0].shape[0]/max_height
left,   width  = 0, X[0].shape[1]/max_width
##ax1 = plt.axes([left, bottom, width, height])
ax1.imshow(X[0]/255)
ax1.set_position([left, bottom, width, height-0.05])
ax1.set_title("X[0]: {}".format(X[0].shape[:2]))
ax1.axis("off")

# output[0] display
bottom2, height2 =  0.01,    output[0].shape[0]/max_height
left2,   width2  = left+width, output[0].shape[1]/max_width
##ax2 = plt.axes([left2, bottom2, width2, height2])
ax2.imshow(output[0]/255)
ax2.set_position([left2, bottom2, width2, height2-0.05])
ax2.set_title("output[0]: {}".format(output[0].shape[:2]))
ax2.axis("off")
plt.show()
