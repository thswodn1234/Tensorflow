import tensorflow as tf
import numpy as np

y = np.arange(10) # integer label
print("y=", y)

y1 = tf.keras.utils.to_categorical(y) # keras one-hot label
print("y1=", y1)

##y2 = tf.one_hot(y, depth=10) # tensorflow one-hot label
##print("y2=", y2.numpy())
