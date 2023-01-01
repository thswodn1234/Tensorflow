'''
ref1:  
https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
ref2: # https://keras.io/applications/
'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
#from tensorflow.keras.applications.imagenet_utils  import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image # pip install pillow

#1:
##import tensorflow as tf
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)

#2:  
##W = 'C:/Users/user/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
model = VGG16(weights='imagenet', include_top=True) # weights= W
model.summary()

#3: predict an image
img_path = './data/elephant.jpg' # './data/dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
x = preprocess_input(x) # mode='caffe'
output = model.predict(x)

#3-1: (class_name, class_description, score)
print('Predicted:', decode_predictions(output, top=5)[0])


#3-2: direct Top-1, and Top-5
k = np.argmax(output[0])            # top 1
z = output[0].argsort()[-5:][::-1]    # top 5

# Imagenet 1000 labels
labels = {}
name = "./DATA/imagenet1000_clsidx_to_labels.txt"
with open(name, 'r') as f:
    C = [line[:-2] for line in f.readlines()]
C[0] = C[0][1:]
for line in C:
    line = line.replace("'", "")
    key, value = line.split(':')
    labels[int(key)] = value.strip()    
print('Top-1 prediction:', labels[k])
print('Top-5 prediction:', [labels[i] for i in z])

#4: display image and labels
plt.imshow(img)
plt.title(labels[k])
plt.axis("off")
plt.show()
