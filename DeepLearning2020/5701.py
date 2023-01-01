'''
ref1: https://github.com/AndreyTulyakov/Simple-U-net-Example
ref2: https://www.tensorflow.org/tutorials/images/segmentation
'''
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate 
from tensorflow.keras.layers import Dense,  Flatten, Reshape, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, UpSampling2D

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image # pip install pillow

import numpy as np
import matplotlib.pyplot as plt
import os

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: 
def load_oxford_pets_2(target_size= (128, 128), test_split_rate= 0.2):
    input_file = "./Oxford_Pets/annotations/list.txt"
    file = open(input_file)
    list_txt = file.readlines()
    file.close()
    
    list_txt = list_txt[6:]     # delete header
    np.random.shuffle(list_txt)

    # load dataset
    dataset = {"name": [], "label": [], "image": [ ], "mask": [] }
    for line in list_txt:
        image_name, class_id, species,  breed_id = line.split()
        image_file= "./Oxford_Pets/images/"+ image_name + ".jpg"
        mask_file = "./Oxford_Pets/annotations/trimaps/"+ image_name + ".png"

        if os.path.exists(image_file) and os.path.exists(mask_file):
            dataset["name"].append(image_name)
            dataset["label"].append(int(species)-1) # Cat: 0, Dog: 1

            # read image and scale to target_size
            img = image.load_img(image_file, target_size=target_size)
            img = image.img_to_array(img)  # (128, 128, 3)
            dataset["image"].append(img)

            # read mask
            mask = image.load_img(mask_file, target_size= target_size,
                                             color_mode='grayscale')    
            mask = image.img_to_array(mask) # (128, 128, 1)
            dataset["mask"].append(mask)

    # change list to np.array
    dataset["name"]  = np.array(dataset["name"])
    dataset["label"] = np.array(dataset["label"])
    dataset["image"] = np.array(dataset["image"])    
    dataset["mask"]  = np.array(dataset["mask"])
    
    # split dataset into train_dataset and test_dataset
    dataset_total = dataset['image'].shape[0]
    test_size     = int(dataset_total*test_split_rate)
    train_size    = dataset_total - test_size

    train_dataset = {}
    train_dataset["name"]= dataset["name"][:train_size]
    train_dataset["label"]= dataset["label"][:train_size]
    train_dataset["image"]= dataset["image"][:train_size]
    train_dataset["mask"]= dataset["mask"][:train_size]
    
    test_dataset  = {}    
    test_dataset["name"] = dataset["name"][train_size:]
    test_dataset["label"]= dataset["label"][train_size:]
    test_dataset["image"]= dataset["image"][train_size:]
    test_dataset["mask"]= dataset["mask"][train_size:]     
    return train_dataset, test_dataset

train_dataset, test_dataset = load_oxford_pets_2() # target_size= (128, 128)
print("train_dataset['image'].shape=", train_dataset['image'].shape)# (5880, 128, 128, 3)
print("test_dataset['image'].shape=",  test_dataset['image'].shape) # (1469, 128, 128, 3)

x_train = train_dataset["image"]/255.0
x_test = test_dataset["image"]/255.0

y_train = train_dataset["mask"]-1 # [1, 2, 3] -> [0, 1, 2]
y_test = test_dataset["mask"]-1
print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)

#3:
def unet_1(input_shape=(128, 128, 3), num_classes=3):
  
    inputs = Input(shape=input_shape)
    # 128
    
    down1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Conv2D(32, (3, 3), activation='relu', padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1_pool = MaxPool2D()(down1)
    # 64

    center = Conv2D(512, (3, 3), activation='relu', padding='same')(down1_pool)
    center = BatchNormalization()(center)
    center = Conv2D(512, (3, 3), activation='relu', padding='same')(center)
    center = BatchNormalization()(center)
    # center

    up1 = UpSampling2D((2, 2))(center)
    up1 = concatenate([down1, up1], axis=3) # try comment this line, i.e, without this shotcut
    up1 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    # 128

    classify = Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(up1)   
    model = Model(inputs=inputs, outputs=classify)
    
    model.compile(optimizer=RMSprop(0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])    
    return model

#4:
UNET = unet_1()
##UNET.summary()
ret = UNET.fit(x_train, y_train, epochs = 20, batch_size = 8, verbose = 2)
'''message:
tensorflow.python.framework.errors_impl.ResourceExhaustedError:  OOM when allocating tensor

solution: reduce batch_size to  8, 4, 2...
'''
train_loss, train_acc = UNET.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc   = UNET.evaluate(x_test,  y_test, verbose=2)

#5:
def display(display_list):
  plt.figure(figsize=(12, 4))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()
  
#6:  
def create_mask(pred_mask):  # (:, 128, 128, 3)
  pred_mask = tf.argmax(pred_mask, axis=-1) # (:, 128, 128), axis=3 
  pred_mask = pred_mask[..., tf.newaxis]    # (:, 128, 128, 1) 
  return pred_mask

# predict segmentation of train data
k = 2
pred_mask = UNET.predict(x_train[:k])  # pred_mask.shape = (k, 128, 128, 3)
pred_mask = create_mask(pred_mask)     # TensorShape([k, 128, 128, 1])

for i in range(k):
    display([x_train[i], y_train[i], pred_mask[i]])

#7: predict segmentation of test data
pred_mask = UNET.predict(x_test[:k])  # pred_mask.shape = (k, 128, 128, 3)
pred_mask = create_mask(pred_mask)    # TensorShape([k, 128, 128, 1])

for i in range(k):
    display([x_test[i], y_test[i], pred_mask[i]])    
