import tensorflow as tf
import numpy as np
#1: crate a 2D input data
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]).astype('float32')
#2
pads = np.array([[1, 1],
                     [2, 2]])
#3
B = tf.pad(A, pads, "constant")
C = tf.pad(A, pads, "reflect")
D = tf.pad(A, pads, "symmetric")
print("B=", B.numpy())
print("C=", C.numpy())
print("D=", D.numpy())
