import tensorflow as tf
import numpy as np

SCE = tf.keras.losses.SparseCategoricalCrossentropy()

t = tf.convert_to_tensor([0, 1, 2, 3])
y =tf.convert_to_tensor([[0.4, 0.3, 0.2, 0.1], #y[0]
                         [0.1, 0.3, 0.2, 0.4]])#y[1]

#1
print("SCE(t[i], y[0])")
print("SCE(t[0], y[0])=", SCE(t[0], y[0]).numpy() ) 
print("SCE(t[1], y[0])=", SCE(t[1], y[0]).numpy() )
print("SCE(t[2], y[0])=", SCE(t[2], y[0]).numpy() )
print("SCE(t[3], y[0])=", SCE(t[3], y[0]).numpy() ) 

#2
print("SCE(t[i], y[1])")
print("SCE(t[0], y[1])=", SCE(t[0], y[1]).numpy() ) 
print("SCE(t[1], y[1])=", SCE(t[1], y[1]).numpy() )
print("SCE(t[2], y[1])=", SCE(t[2], y[1]).numpy() )
print("SCE(t[3], y[1])=", SCE(t[3], y[1]).numpy() )

#3
print("SCE(tf.stack((t[1], t[1])), y)=",
       SCE(tf.stack((t[1], t[1])), y).numpy() )
