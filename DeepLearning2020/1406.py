import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 # pip install opencv-contrib-python

def dataset(train_size=100): # tensorflow    
     tf.random.set_seed(1)
     x = tf.linspace(-5.0, 5.0, num=train_size)
     y = 3.0*x**3 + 2.0*x**2 + x + 4.0
     y += tf.random.normal([train_size], mean=0.0, stddev = 30.0)
     return x, y
x, y_true = dataset(20)

# n-차 다항식 회귀
n = 3
X = np.ones(shape = (len(x), n+1), dtype=np.float32)
for i in range(1, n+1):
     X[:, i] = x**i

#텐서플로 모델, 학습결과 로드
fname = "./RES/frozen_graph.pb"
net =cv2.dnn.readNetFromTensorflow(fname)
##net =cv2.dnn.readNetFromTensorflow(np.fromfile(fname, dtype=np.uint8))#한글 path
##for xx in X:
##    blob = cv2.dnn.blobFromImage(xx)
##    net.setInput(blob)
##    res = net.forward()
##    print(xx, res)
     
blob = cv2.dnn.blobFromImages(X) # blob.shape = (20, 1, 4, 1)
net.setInput(blob) 
y_pred = net.forward()

plt.scatter(x, y_true) 
plt.plot(x, y_pred, color='red')
plt.show()
