'''
ref1: https://keras.io/applications/
ref2:https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py
ref3:
https://github.com/keras-team/keras-applications/releases/tag/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels.h5
'''
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input,decode_predictions 
from tensorflow.keras.preprocessing import image # pip install pillow
import numpy as np
import matplotlib.pyplot as plt

#1:
##import tensorflow as tf
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)

#2:
##W = 'C:/Users/user/.keras/models/resnet50v2_weights_tf_dim_ordering_tf_kernels.h5'
model = ResNet50V2(weights='imagenet', include_top=True) # weights= W
model.summary()

#3: predict an image
#3-1:
img_path = './data/elephant.jpg' # './data/dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) # (1, 224, 224, 3)
x = preprocess_input(x)
##x = tf.keras.applications.imagenet_utils.preprocess_input(x, mode='tf')
output = model.predict(x)

#3-2: (class_name, class_description, score)
top5 = decode_predictions(output, top=5)[0]
print('Top-5 predicted:', top5)
#direct Top-1, and Top-5, ref[4502]

#4: display image and labels  
plt.imshow(img)
plt.title(top5[0][1])
plt.axis("off")
plt.show()
