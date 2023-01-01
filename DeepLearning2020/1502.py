import tensorflow as tf
import numpy as np

CCE = tf.keras.losses.CategoricalCrossentropy()
t= np.array([[1,   0,   0,   0],   #t[0]
             [0,   1,   0,   0],   #t[1]
             [0,   0,   1,   0],   #t[2]
             [0,   0,   0,   1]])  #t[3]
 
y =np.array([[0.4, 0.3, 0.2, 0.1], #y[0]
             [0.1, 0.3, 0.2, 0.4]])#y[1]
#1
print("CCE(t[i], y[0])")
print("CCE(t[0], y[0])=", CCE(t[0], y[0]).numpy() ) 
print("CCE(t[1], y[0])=", CCE(t[1], y[0]).numpy() )
print("CCE(t[2], y[0])=", CCE(t[2], y[0]).numpy() )
print("CCE(t[3], y[0])=", CCE(t[3], y[0]).numpy() ) 

#2
print("CCE(t[i], y[1])")
print("CCE(t[0], y[1])=", CCE(t[0], y[1]).numpy() ) 
print("CCE(t[1], y[1])=", CCE(t[1], y[1]).numpy() )
print("CCE(t[2], y[1])=", CCE(t[2], y[1]).numpy() )
print("CCE(t[3], y[1])=", CCE(t[3], y[1]).numpy() )

#3
print("CCE(np.vstack((t[1], t[1])), y)=",
       CCE(np.vstack((t[1], t[1])), y).numpy() )
