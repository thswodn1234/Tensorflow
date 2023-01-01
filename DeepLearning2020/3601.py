import tensorflow as tf
import numpy as np
#1: crate a 1D input data
A = np.array([1, 2, 3, 4, 5]).astype('float32')

#2
p = 2
paddings = np.array([[p, p]])

#3
B = tf.pad(A, paddings, "constant")
C = tf.pad(A, paddings, "reflect")
D = tf.pad(A, paddings, "symmetric")
print("B=", B.numpy())
print("C=", C.numpy())
print("D=", D.numpy())
