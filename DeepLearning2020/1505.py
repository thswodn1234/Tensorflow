import tensorflow as tf
import numpy as np

BCE = tf.keras.losses.BinaryCrossentropy()
t= np.array([[1,   1,   0,   0],   #t[0]
             [0,   1,   1,   0],   #t[1]
             [0,   0,   1,   1],   #t[2]
             [0,   1,   0,   1]])  #t[3]

y =np.array([[0.4, 0.3, 0.2, 0.1], #y[0]
             [0.1, 0.3, 0.2, 0.4]])#y[1]
#1
print("BCE(t[i], y[0])")
print("BCE(t[0], y[0])=", BCE(t[0], y[0]).numpy() ) 
print("BCE(t[1], y[0])=", BCE(t[1], y[0]).numpy() )
print("BCE(t[2], y[0])=", BCE(t[2], y[0]).numpy() )
print("BCE(t[3], y[0])=", BCE(t[3], y[0]).numpy() ) 

#2
print("BCE(t[i], y[1])")
print("BCE(t[0], y[1])=", BCE(t[0], y[1]).numpy() ) 
print("BCE(t[1], y[1])=", BCE(t[1], y[1]).numpy() )
print("BCE(t[2], y[1])=", BCE(t[2], y[1]).numpy() )
print("BCE(t[3], y[1])=", BCE(t[3], y[1]).numpy() )

#3
print("BCE(np.vstack((t[0], t[0])), y)=",
       BCE(np.vstack((t[0], t[0])), y).numpy() )
