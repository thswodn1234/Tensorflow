import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import datetime
import io

#1
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train.shape=", x_train.shape)
print("x_test.shape=",  x_test.shape)

#2
import os
path = "c:\\tmp\\logs\\"
if not os.path.isdir(path):
    os.mkdir(path)
##logdir = path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = path + "3103"

#3
file_writer = tf.summary.create_file_writer(logdir + "/train")
file_writer.set_as_default()
img = np.reshape(x_train[0:4], (-1, 28, 28, 1)) # NHWC= (4, 28, 28, 1)
tf.summary.image("x_train", img, max_outputs=4, step=0)
tf.summary.flush()

#4
file_writer= tf.summary.create_file_writer(logdir + "/test")
with file_writer.as_default():
    img = np.reshape(x_test[0:4], (-1, 28, 28, 1)) # NHWC= (4, 28, 28, 1)
    tf.summary.image("x_test", img, max_outputs=4, step=0)
    tf.summary.flush()

#5, ref: https://www.tensorflow.org/tensorboard/image_summaries
def plot_to_image(figure):
  buf = io.BytesIO()
  plt.savefig(buf, format='png') # Save the plot to a PNG in memory
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4) # HWC=(H,W,4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0) # NHWC = (1, H, W, 4)
  return image

#6: draw images at the figure of matplotlib
fig = plt.figure(figsize=(8, 4))
for i in range(8):
    plt.subplot(2, 4, i + 1)  
    plt.imshow(x_train[i], cmap='gray')
    plt.axis("off")
fig.tight_layout()

#7: write plt to tensorboard using plot_to_image()
file_writer= tf.summary.create_file_writer(logdir + "/matplotlib")
with file_writer.as_default():
    tf.summary.image("train", plot_to_image(fig), step=0)
