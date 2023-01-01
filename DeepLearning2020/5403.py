'''
ref1: https://www.tensorflow.org/tutorials/generative/dcgan?hl=ko
ref2: https://github.com/Zackory/Keras-MNIST-GAN/blob/master/mnist_dcgan.py
'''
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, LeakyReLU, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2:
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')/127.5 - 1.0  # [ -1, 1]
x_train = np.expand_dims(x_train, axis=3)        # (60000, 28, 28, 1)

opt = tf.keras.optimizers.RMSprop(learning_rate = 0.0002)
##opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1 = 0.5)

##init_lr = 0.0002
##lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
##              init_lr, decay_steps=469*10*2, decay_rate=0.96, staircase=True)
##opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

#3: create model 
#3-1: generator, G
noise_dim = 100
g_input = Input(shape = (noise_dim, ))
x= Dense(units = 7*7*128, activation = 'relu')(g_input)
x= Reshape((7, 7, 128))(x)
x= Conv2DTranspose(filters=64, kernel_size = (3, 3), strides = (2, 2), 
                   activation = 'relu', padding = 'same')(x)
x= BatchNormalization()(x)
x= Conv2DTranspose(filters=32, kernel_size = (3, 3), strides = (2, 2), 
                   activation = 'relu', padding = 'same')(x)
x= BatchNormalization()(x)

g_output= Conv2D(filters=1, kernel_size = (3, 3), strides = (1, 1), 
              activation = 'tanh', padding = 'same')(x) # (None, 28, 28, 1)
G= Model(inputs= g_input, outputs= g_output, name ='G')
G.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
G.summary()

#3-2: discriminator, D
d_input = Input(shape = (28, 28, 1))
x= Conv2D(32, kernel_size=3, strides=2, padding="same")(d_input)
x= LeakyReLU()(x)
x= Dropout(0.3)(x)

x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x) 
x= LeakyReLU()(x)
x= Dropout(0.3)(x)

x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x) 
x= LeakyReLU()(x)
x= Dropout(0.3)(x)

x = Flatten()(x)
d_output = Dense(1, activation='sigmoid')(x)

D = Model(inputs= d_input, outputs= d_output, name="D")
D.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
D.summary()

#3-3: GAN model
D.trainable = False 
gan_input = Input(shape=(noise_dim,))
DCGAN = Model(inputs=gan_input, outputs=D(G(gan_input)), name="GAN")
DCGAN.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
DCGAN.summary()

#4:
import os
if not os.path.exists("./GAN"):
     os.mkdir("./GAN")

def plotGeneratedImages(epoch, examples=20, dim=(2, 10), figsize=(10, 2)):
    noise = np.random.normal(0, 1, size=[examples, noise_dim])
    g_image = G.predict(noise)
    g_image = np.squeeze(g_image, axis = 3) 

    g_image = (g_image + 1.0)*127.5
    g_image = g_image.astype('uint8')

    plt.figure(figsize=figsize)
    for i in range(g_image.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(g_image[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("./GAN/dcgan_epoch_%d.png"% epoch)
    plt.close()
      
#5:
BUFFER_SIZE = x_train.shape[0] # 60000
BATCH_SIZE  = 128
batch_count = np.ceil(BUFFER_SIZE/BATCH_SIZE)
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

history = {"g_loss":[], "g_acc":[], "d_loss":[], "d_acc":[]}
def train(epochs=100):

    for epoch in range(epochs):
        dloss = 0.0
        gloss = 0.0
        dacc  = 0.0
        gacc  = 0.0

##      batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
##      print("epoch = ", D.optimizer._decayed_lr('float32').numpy())

        for batch in train_dataset:  # batch.shape = (BATCH_SIZE, 28, 28, 1)
            batch_size = batch.shape[0]
            
            noise = tf.random.normal([batch_size, noise_dim])
            fake = G.predict(noise)  # fake.shape = (batch_size, 784)
            X = np.concatenate([batch, fake]) # X.shape = (2*batch_size, 784)     

            # labels for fake = 0, batch = 1
            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size] = 1.0

            # train discriminator, D
            ret = D.train_on_batch(X, y_dis) # D.trainable = True
            dloss += ret[0] # loss
            dacc  += ret[1] # accuracy
            
            # train generator, G
            noise = tf.random.normal([batch_size, noise_dim])
            y_gen = np.ones(batch_size)
            ret= DCGAN.train_on_batch(noise, y_gen) # D.trainable = False
            gloss += ret[0]
            gacc  += ret[1]

        avg_gloss = gloss/batch_count
        avg_gacc  = gacc/batch_count
        
        avg_dloss = dloss/batch_count
        avg_dacc  = dacc/batch_count

    
        print("epoch={}: G:(loss= {:.4f}, acc={:.1f}), D:(loss= {:.4f}, acc={:.1f})".format(
            epoch, avg_gloss,100*avg_gacc, avg_dloss, 100*avg_dacc))
        history["g_loss"].append(avg_gloss)
        history["g_acc"].append(avg_gacc)
        history["d_loss"].append(avg_dloss)
        history["d_acc"].append(avg_dacc)
   
        if epoch % 20 == 0 or epoch == epochs-1:
            plotGeneratedImages(epoch)
train(100)
   
#6:
fig, ax = plt.subplots(1, 2, figsize = (10, 6))
ax[0].plot(history["g_loss"], "g-", label = "G losses")
ax[0].plot(history["d_loss"], "b-", label = "D losses")
ax[0].set_title("train loss")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")
ax[0].legend()

ax[1].plot(history["g_acc"], "g-",  label = "G accuracy")
ax[1].plot(history["d_acc"], "b-",  label = "D accuracy")
ax[1].set_title("accuracy")
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("accuracy")
ax[1].legend()
fig.tight_layout()
plt.show()
