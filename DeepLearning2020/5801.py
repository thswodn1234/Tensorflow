'''
ref1: http://www.robots.ox.ac.uk/~vgg/data/pets/
ref2: https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Localizer.ipynb?pli=1#scrollTo=aEXMx4KAkN6d
'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image # pip install pillow
import os

import xml.etree.ElementTree as ET
import cv2  # pip install opencv_python

import tensorflow as tf
from tensorflow.keras.layers   import Input, Dense, Flatten
from tensorflow.keras.layers   import Conv2D, BatchNormalization,MaxPool2D, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#2: extract Bounding Box from xml
def getBB(file_path):
  try:
    tree = ET.parse(file_path)
  except FileNotFoundError:
    return None  
  root = tree.getroot()  
  ob = root.find('object')
  bndbox = ob.find('bndbox')
  xmin = bndbox.find('xmin').text
  xmax = bndbox.find('xmax').text
  ymin = bndbox.find('ymin').text
  ymax = bndbox.find('ymax').text
  return [int(xmin), int(ymin), int(xmax), int(ymax)]

def load_oxford_pets_3(target_size= (224, 224)):
    input_file = "./Oxford_Pets/annotations/list.txt"
    file = open(input_file)
    list_txt = file.readlines()
    file.close()
    
    list_txt = list_txt[6:]     # delete header
    np.random.shuffle(list_txt)

    # load dataset
    train_dataset= {"name": [], "label": [], "image": [ ], "box": [] }
    test_dataset = {"name": [], "label": [], "image": [ ]}

    for line in list_txt:    
        image_name, class_id, species, breed_id = line.split()
        image_file= "./Oxford_Pets/images/"+ image_name + ".jpg"
        box_file  = "./Oxford_Pets/annotations/xmls/"+ image_name + ".xml"     

        if not os.path.exists(image_file):
            continue

        # read image and scale to target_size
        img = image.load_img(image_file) # read as original size
        sx = target_size[0]/img.width    # for rescaling BB
        sy = target_size[1]/img.height
            
        img = img.resize(size=target_size)
        img = image.img_to_array(img)  # (224, 224, 3)
          
        if  os.path.exists(box_file): # train_dataset
            # read xml, rescale box by target_size   
            box = getBB(box_file)
            box[0] = round(box[0]*sx) # scale xmin with sx
            box[1] = round(box[1]*sy) # scale ymin with sy
            box[2] = round(box[2]*sx) # scale xmax with sx
            box[3] = round(box[3]*sy) # scale ymax with sy
            train_dataset["box"].append(box)
            train_dataset["name"].append(image_name)
            train_dataset["label"].append(int(species)-1)
            train_dataset["image"].append(img)          
            
        else: #test_dataset
            test_dataset["name"].append(image_name)
            test_dataset["label"].append(int(species)-1)
            test_dataset["image"].append(img)        
    # change list to np.array
    train_dataset["image"] = np.array(train_dataset["image"])
    train_dataset["box"]  = np.array(train_dataset["box"])
    train_dataset["label"] = np.array(train_dataset["label"])
    train_dataset["name"]  = np.array(train_dataset["name"])

    test_dataset["image"] = np.array(test_dataset["image"])
    test_dataset["label"] = np.array(test_dataset["label"])
    test_dataset["name"]  = np.array(test_dataset["name"])    
    return train_dataset, test_dataset
 
train_dataset, test_dataset = load_oxford_pets_3()
print("train_dataset['image'].shape=", train_dataset['image'].shape)# (5880, 224, 224, 3)

#normalize 
x_train = train_dataset["image"]/255.0
y_train = train_dataset["box"]/x_train.shape[1] # [0, 224] -> [0, 1]
x_test  = test_dataset["image"]/255.0

#3:
def IOU(y_true, y_pred):   

    b1_xmin, b1_ymin, b1_xmax, b1_ymax = tf.unstack(y_true, 4, axis=-1)
    b2_xmin, b2_ymin, b2_xmax, b2_ymax = tf.unstack(y_pred, 4, axis=-1)

    zero = tf.convert_to_tensor(0.0, y_true.dtype) # zero = 0.0      
    b1_width  = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width  = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)

    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area/union_area # tf.math.divide_no_nan(intersect_area, union_area)
    return iou

#4: build a cnn model
def create_cnn2d(input_shape, num_units = 4):
    inputs = Input(shape = input_shape)
    x= Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu')(inputs)
    x= BatchNormalization()(x)
    x= MaxPool2D()(x)

    x= Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu')(x)
    x= BatchNormalization()(x)
    x= MaxPool2D()(x)
    x= Dropout(rate = 0.2)(x)

    x= Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu')(x)
    x= BatchNormalization()(x)    
    x= MaxPool2D()(x)
    x= Dropout(rate = 0.2)(x)
    
    x = Flatten()(x)
    x = Dense(128, activation = 'relu')(x)
    x= BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(units = num_units, activation = 'sigmoid')(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model
model = create_cnn2d(input_shape = x_train.shape[1:])

#5: train the model
opt = RMSprop(learning_rate = 0.001)
model.compile(optimizer = opt, loss = 'mse', metrics = [IOU])
ret = model.fit(x_train, y_train, epochs =100, batch_size =128, verbose = 0)

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)

#6: plot accuracy and loss
fig, ax = plt.subplots(1, 2, figsize = (10, 6))
ax[0].plot(ret.history['loss'], "g-")
ax[0].set_title('train loss')
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

ax[1].plot(ret.history['IOU'], "b-")
ax[1].set_title('IOU')
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('IOU')
fig.tight_layout()
plt.show()

#7: predict k samples and display results
k = 8
train_box = model.predict(x_train[:k])
test_box = model.predict(x_test[:k])

def display_images(img, pred_box, true_box=None, size= 224):
    box = pred_box*size
    box = box.astype(int)
    k =  pred_box.shape[0]
    fig = plt.figure(figsize=(8, k//2))

    for i in range(k):
        plt.subplot(k//4, 4, i+1)

        a_img = (img[i]*255).astype('uint8')

        # box predicted
        xmin, ymin, xmax, ymax = box[i]           
        cv2.rectangle(a_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)

        if true_box is not None: # true box in case of train data
            xmin, ymin, xmax, ymax = true_box[i]
            cv2.rectangle(a_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 5)
        plt.imshow(a_img)
        plt.axis("off")
    fig.tight_layout()
    plt.show()

display_images(x_train[:k], train_box, train_dataset["box"][:k])
display_images(x_test[:k],  test_box)
