'''
ref1: http://www.robots.ox.ac.uk/~vgg/data/pets/
ref2: https://github.com/mpecha/Oxford-IIIT-Pet-Dataset
'''
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense  
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image # pip install pillow
import os

#1: 
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)

#2: 
def load_oxford_pets_1(target_size= (224, 224), test_split_rate= 0.2):
    input_file = "./Oxford_Pets/annotations/list.txt"
    file = open(input_file)
    list_txt = file.readlines()
    file.close()
    
    list_txt = list_txt[6:]     # delete header
    np.random.shuffle(list_txt)

    # load dataset
    dataset = {"name": [], "label": [], "image": [ ]}
    for line in list_txt:
        image_name, class_id, species,  breed_id = line.split()
        image_file= "./Oxford_Pets/images/"+ image_name + ".jpg"

        if os.path.exists(image_file):
            dataset["name"].append(image_name)
            dataset["label"].append(int(species)-1) # Cat: 0, Dog: 1

            # read image and scale to target_size
            img = image.load_img(image_file, target_size=target_size)
            img = image.img_to_array(img)  # (224, 224, 3)
            dataset["image"].append(img)

    # change list to np.array
    dataset["image"] = np.array(dataset["image"])
    dataset["label"] = np.array(dataset["label"])
    dataset["name"]  = np.array(dataset["name"])    

    # split dataset into train_dataset and test_dataset
    dataset_total = dataset['image'].shape[0]
    test_size     = int(dataset_total*test_split_rate)
    train_size    = dataset_total - test_size
   
    train_dataset = {} 
    train_dataset["image"]= dataset["image"][:train_size]
    train_dataset["label"]= dataset["label"][:train_size]
    train_dataset["name"]= dataset["name"][:train_size]
    
    test_dataset = {} 
    test_dataset["image"]= dataset["image"][train_size:]
    test_dataset["label"]= dataset["label"][train_size:]
    test_dataset["name"] = dataset["name"][train_size:]
    
    return train_dataset, test_dataset
 
#3
train_dataset, test_dataset = load_oxford_pets_1()
x_train = train_dataset["image"]/255.0
y_train = train_dataset["label"]

x_test = test_dataset["image"]/255.0
y_test = test_dataset["label"]

##print("x_train.shape = ", x_train.shape) # (5880, 224, 224, 3)
##print("y_train.shape = ", y_train.shape) # (5880, )

# one-hot encoding 
y_train = tf.keras.utils.to_categorical(y_train) 
y_test = tf.keras.utils.to_categorical(y_test)

#4: build a functional cnn model
def create_cnn2d(input_shape, num_class = 2):
    inputs = Input(shape = input_shape)
    x= Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu')(inputs)
    x= BatchNormalization()(x)
    x= MaxPool2D()(x)

    x= Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu')(x)
    x= BatchNormalization()(x)
    x= MaxPool2D()(x)
    x= Dropout(rate = 0.5)(x)
    
    x = Flatten()(x)
    outputs = tf.keras.layers.Dense(units = num_class, activation = 'softmax')(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model
model = create_cnn2d(input_shape = x_train.shape[1:])
##model.summary()

#5: train the model
opt = RMSprop(learning_rate = 0.001)
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
ret = model.fit(x_train, y_train, epochs = 50, batch_size =64, verbose = 0)

#6: evaluate the model
train_loss, train_acc = model.evaluate(x_train, y_train, verbose = 2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose = 2)

#7: plot accuracy and loss
fig, ax = plt.subplots(1, 2, figsize = (10, 6))
ax[0].plot(ret.history['loss'], "g-")
ax[0].set_title("train loss")
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

ax[1].plot(ret.history['accuracy'], "b-")
ax[1].set_title("accuracy")
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')
fig.tight_layout()
plt.show()

