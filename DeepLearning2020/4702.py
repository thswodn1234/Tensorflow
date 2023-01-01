'''
ref1: https://keras.io/applications/
ref2: InceptionV3
https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5

ref3: InceptionResNetV2
https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5

ref4: Xception
https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5 
'''
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, Xception
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions 
from tensorflow.keras.preprocessing import image # pip install pillow
import numpy as np
import matplotlib.pyplot as plt

#1:
##import tensorflow as tf
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)


#2:
model1 = InceptionV3(weights='imagenet', include_top=True)
model2 = InceptionResNetV2(weights='imagenet', include_top=True)
model3 = Xception(weights='imagenet', include_top=True)
#model1.summary()
#model2.summary()
#model3.summary()

#3: predict an image
#3-1:
img_path = './data/elephant.jpg' # './data/dog.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) # (1, 299, 299, 3)
x = preprocess_input(x)
output1 = model1.predict(x)
output2 = model2.predict(x)
output3 = model3.predict(x)

#3-2: (class_name, class_description, score)
top5 = decode_predictions(output1, top=5)[0]
print('InceptionV3, Top-5:', top5)

top5 = decode_predictions(output2, top=5)[0]
print('InceptionResNetV2, Top-5:', top5)

top5 = decode_predictions(output3, top=5)[0]
print('Xception, Top-5:', top5)

#4: display image and labels  
##plt.imshow(img)
##plt.title(top5[0][1])
##plt.axis("off")
##plt.show()
