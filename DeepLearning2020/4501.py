'''
ref:
https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
'''

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
##from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.imagenet_utils  import preprocess_input

#1: RGB
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') # (50000, 32, 32, 3)

#2: default mode 'caffe' style, BGR
X = x_train.copy()
Y1 = preprocess_input(X)
del X  # 메모리 삭제
np.set_printoptions(precision=3, threshold=10)
print('Y1[0,0,:,0]=', Y1[0,0,:,0])

#2-1: for checking Y1
X = x_train.copy()                      
X = X[..., ::-1] # RGB to BGR
mean = np.array([103.939, 116.779, 123.68], dtype=np.float32) # ImageNet
Y2 = X-mean      #the same as Y1
del X
print('Y2[0,0,:,0]=', Y2[0,0,:,0])
print("np.allclose(Y1,Y2)=", np.allclose(Y1,Y2, rtol=1e-03))

#3: mode='tf' 
X = x_train.copy()
Y3 = preprocess_input(X, mode='tf')
del X
print('Y3[0,0,:,0]=', Y3[0,0,:,0])
                     
#3-1: for checking Y3
X = x_train.copy()                      
X /= 127.5
Y4 = X - 1.0 # the same as Y3
del X
print('Y4[0,0,:,0]=', Y4[0,0,:,0])
print("np.allclose(Y3, Y4)=",np.allclose(Y3, Y4, rtol=1e-03))
 
#4: mode='torch'
X = x_train.copy()
Y5 = preprocess_input(X, mode='torch')
del X
print('Y5[0,0,:,0]=', Y5[0,0,:,0])

#4-1: for checking Y5
X = x_train.copy()
X /= 255.
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) # ImageNet
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32) # ImageNet
Y6 = (X-mean)/std # the same as Y5
del X
print('Y6[0,0,:,0]=', Y6[0,0,:,0]) 
##print("np.allclose(Y5, Y6)=",np.allclose(Y5, Y6, rtol=1e-03))

#5: display image: x_train[0,:,:,:], Y1[0,:,:,:]
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
##ax[0].imshow(x_train[0,:,:,:]/255.0)
ax[0].imshow(x_train[0,:,:,:].astype(np.uint8))
ax[0].set_title("x_train[0]:RGB")
ax[0].axis("off")

mean = np.array([103.939, 116.779, 123.68], dtype=np.float32) # ImageNet
Y1 +=mean
##ax[1].imshow(Y1[0,:,:,:]/255.0)
ax[1].imshow(Y1[0,:,:,:].astype(np.uint8))
ax[1].set_title("Y1[0]:BGR")
ax[1].axis("off")
fig.tight_layout()
plt.show() 
