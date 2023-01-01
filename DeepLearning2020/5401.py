'''
ref1: https://www.tensorflow.org/tutorials/generative/dcgan?hl=ko
ref2: https://github.com/Zackory/Keras-MNIST-GAN/blob/master/mnist_gan.py
'''
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout
import numpy as np
import matplotlib.pyplot as plt

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2:
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')/127.5 - 1.0  # [ -1, 1]
x_train = x_train.reshape(-1, 784)

#3: G, D using Sequential
noise_dim = 10 # 100

#3-1: generator, G
##G = Sequential()
##G.add(Dense(256, input_dim=noise_dim ))
##G.add(LeakyReLU(alpha=0.2))
##G.add(Dense(512))
##G.add(LeakyReLU(alpha=0.2))
##G.add(Dense(1024))
##G.add(LeakyReLU(alpha=0.2))
##G.add(Dense(784, activation='tanh')) #[-1, 1]
##G.compile(loss='binary_crossentropy', optimizer='rmsprop') 

#3-2:discriminator, D
##D = Sequential()
##D.add(Dense(1024, input_dim=784))
##D.add(LeakyReLU(alpha=0.2))
##D.add(Dropout(0.3))

##D.add(Dense(512))
##D.add(LeakyReLU(alpha=0.2))
##D.add(Dropout(0.3))
##D.add(Dense(256))
##D.add(LeakyReLU(alpha=0.2))
##D.add(Dropout(0.3))
##D.add(Dense(1, activation='sigmoid'))
##D.compile(loss='binary_crossentropy', optimizer='rmsprop')

#4: G, D using Model
noise_dim = 10 # 100
#4-1
g_input = Input(shape = (noise_dim, ))
x= Dense(units = 256)(g_input)
x= LeakyReLU(alpha=0.2)(x)
x= Dense(units = 512)(x)
x= LeakyReLU(alpha=0.2)(x)
x= Dense(units = 1024)(x)
x= LeakyReLU(alpha=0.2)(x)
g_out= Dense(784, activation='tanh')(x) # [-1, 1]
G = Model(inputs= g_input, outputs= g_out, name="G")
G.summary()
##G.compile(loss='binary_crossentropy', optimizer='rmsprop')


#4-2: discriminator, D
d_input = Input(shape = (784, ))
x= Dense(units = 1024)(d_input)
x= LeakyReLU()(x)
x= Dropout(0.3)(x)

x= Dense(units = 512)(x)
x= LeakyReLU()(x)
x= Dropout(0.3)(x)

x= Dense(units = 256)(x)
x= LeakyReLU()(x)
x= Dropout(0.3)(x)
d_out= Dense(1, activation='sigmoid')(x)
D = Model(inputs= d_input, outputs= d_out, name="D")
D.summary()
##D.compile(loss='binary_crossentropy', optimizer='rmsprop')

#5: GAN
##D.trainable = False
gan_input = Input(shape=(noise_dim,))
x = G(gan_input)
gan_output = D(x)
GAN = Model(inputs=gan_input, outputs=gan_output, name="GAN")
GAN.summary()
##gan.compile(loss='binary_crossentropy', optimizer='rmsprop')

#6
batch_size = 4
noise = tf.random.normal([batch_size, noise_dim])
fake = G(noise)
out = D(fake)   # out= D(G(noise)), GAN(noise), out= GAN.predict(noise)
print('out=', out)
##print('GAN(noise)=', GAN(noise))
##print('D(x_train[:batch_size])=', D(x_train[:batch_size]))

fig = plt.figure(figsize = (8, 2))
for i in range(batch_size):
    plt.subplot(1, 4, i + 1)  
    plt.imshow(fake[i].numpy().reshape((28, 28)), cmap = 'gray')
    plt.axis("off")
fig.tight_layout()
plt.show()
