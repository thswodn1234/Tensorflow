'''
ref1: https://www.youtube.com/watch?v=GSwYGkTfOKk (C4W3L01: Object Localization by Andrew Ng)
ref2: https://arxiv.org/pdf/1506.02640.pdf (YOLO ver1)
ref3: https://mlblr.com/includes/mlai/index.html#yolov2
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
from tensorflow.keras.layers   import LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16

#1: 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
    
#2: load Oxford_pets dataset
def getBB(file_path):# extract Bounding Box from xml
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
    train_dataset["label"] = np.array(train_dataset["label"]) - 1 #[1, 2] -> [0, 1] 
    train_dataset["name"]  = np.array(train_dataset["name"])

    test_dataset["image"] = np.array(test_dataset["image"])
    test_dataset["label"] = np.array(test_dataset["label"]) - 1   #[1, 2] -> [0, 1] 
    test_dataset["name"]  = np.array(test_dataset["name"])    
    return train_dataset, test_dataset
 
train_dataset, test_dataset = load_oxford_pets_3()
print("train_dataset['image'].shape=", train_dataset['image'].shape)# (5880, 224, 224, 3)
x_train = train_dataset["image"]/255.0  #normalize 
x_test  = test_dataset["image"]/255.0

#3: make labels [pc, x, y, w, h, c1, c2]
def make_labels_oxford_pets(dataset, train_data=True):

    label = dataset["label"]
    N = dataset['image'].shape[0]
    
    if train_data: # normalize box and (x, y, w, h)   
        size = dataset['image'].shape[1] # 224
        box   = dataset["box"]

        x = (box[:,0] + box[:,2])/(2.*size)
        y = (box[:,1] + box[:,3])/(2.*size)
        w = (box[:,2] - box[:,0])/size
        h = (box[:,3] - box[:,1])/size
    else: # no box info, so (0.5,0.5) - (1, 1)              
        x = np.full(shape = (N,), fill_value=0.5, dtype="float32")
        y = np.full(shape = (N,), fill_value=0.5, dtype="float32")
        w = np.ones(shape = (N,), dtype="float32")
        h = np.ones(shape = (N,), dtype="float32")                

    pc = np.ones(shape = (N,), dtype="float32") # all images has an object
        
    C = np.zeros((N, 2))
    C[np.arange(N), label]= 1 # one-hot

    label_y = np.zeros(shape=(N, 7), dtype="float32")#[pc, x, y, w, h, c1, c2]        
    label_y[:, 0] = pc
    label_y[:, 1] = x
    label_y[:, 2] = y
    label_y[:, 3] = w
    label_y[:, 4] = h
    label_y[:, 5:]= C        
    return label_y
       
y_train = make_labels_oxford_pets(train_dataset)
y_test  = make_labels_oxford_pets(test_dataset, train_data=False)

#4:
def IOU(y_true, y_pred):

    b1_pc, b1_x, b1_y, b1_w, b1_h, b1_c1, b1_c2 = tf.unstack(y_true, 7, axis=-1)
    b2_pc, b2_x, b2_y, b2_w, b2_h, b2_c1, b2_c2 = tf.unstack(y_pred, 7, axis=-1)
   
    zero = tf.convert_to_tensor(0.0, y_true.dtype) # zero = 0.0   
    b1_width  = tf.maximum(zero, b1_w)
    b1_height = tf.maximum(zero, b1_h)
    b2_width  = tf.maximum(zero, b2_w)
    b2_height = tf.maximum(zero, b2_h)

    b1_w2 = b1_width/2
    b1_h2 = b1_height/2
    b1_xmin = b1_x - b1_w2
    b1_ymin = b1_y - b1_h2
    b1_xmax = b1_x + b1_w2
    b1_ymax = b1_y + b1_h2
    
    b2_w2 = b2_width/2
    b2_h2 = b2_height/2
    b2_xmin = b2_x - b2_w2
    b2_ymin = b2_y - b2_h2
    b2_xmax = b2_x + b2_w2
    b2_ymax = b2_y + b2_h2
    
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
    iou = intersect_area/union_area
    return iou

#5: 
##@tf.function
def custom_loss(y_true, y_pred): # [pc, x, y, w, h, c1, c2]
    y_true_conf, b1_x, b1_y, b1_w, b1_h, b1_c1, b1_c2 = tf.unstack(y_true, 7, axis=-1)
    y_pred_conf, b2_x, b2_y, b2_w, b2_h, b2_c1, b2_c2 = tf.unstack(y_pred, 7, axis=-1)

##  loss_conf = tf.square(y_true_conf- y_pred_conf)    
    iou = IOU(y_true, y_pred)
    loss_conf = tf.square(y_true_conf*iou- y_pred_conf)

    loss_xy = tf.square(b1_x - b2_x) + tf.square(b1_y - b2_y)    

    b1_w = tf.sqrt(b1_w)
    b1_h = tf.sqrt(b1_h)  
    b2_w = tf.sqrt(b2_w)
    b2_h = tf.sqrt(b2_h)  
    loss_wh = tf.square(b1_w -b2_w) + tf.square(b1_h - b2_h)
    
##    loss_class = tf.square(b1_c1 - b2_c1) + tf.square(b1_c2 - b2_c2)
    
    #categorical cross entropy
    epsilon=1e-12
    b2_c1 = tf.keras.backend.clip(b2_c1, epsilon, 1.0-epsilon)
    b2_c2 = tf.keras.backend.clip(b2_c2, epsilon, 1.0-epsilon)           
    loss_class = -(b1_c1*tf.math.log(b2_c1) + b1_c2*tf.math.log(b2_c2))
    
    # loss sum
    loss = loss_conf + (loss_xy+ loss_wh + loss_class)*y_true_conf
    
    loss = tf.reduce_mean(loss, axis=-1)       
    return loss

##def custom_loss(y_true, y_pred):# [pc, x, y, w, h, c1, c2]
##
##    y_true_conf = y_true[:,0]
##    
##    iou = IOU(y_true, y_pred)   
##    loss_conf = tf.keras.losses.mean_squared_error(y_true_conf*iou, y_pred[:,0]) 
##    loss_xy = tf.keras.losses.mean_squared_error(y_true[:,1:3], y_pred[:,1:3])
##    loss_wh = tf.keras.losses.mean_squared_error(tf.sqrt(y_true[:,3:5]), tf.sqrt(y_pred[:,3:5]))   
##    loss_class = tf.keras.losses.categorical_crossentropy(y_true[...,-2:], y_pred[...,-2:])
##    loss = loss_conf + (loss_xy+ loss_wh + loss_class)*y_true_conf       
##    return loss

#6: 
def custom_acc(y_true, y_pred):
    y_true_class = y_true[...,-2:]
    y_pred_class = y_pred[...,-2:]
    return tf.keras.metrics.categorical_accuracy(y_true_class, y_pred_class)   
    
##    y_true_class = tf.argmax(y_true_class, axis=-1)
##    y_pred_class = tf.argmax(y_pred_class, axis=-1)
##    acc =  tf.cast( tf.math.equal(  y_true_class, y_pred_class ), tf.float32 )
##    return acc # you not need to dvide by total 
    
#7: build a cnn model
#7-1:
def custom_activations(x): # sigmoid(pc, x, y, w, h), softmat(c1, c2)
    x_0 = tf.keras.activations.sigmoid(x[...,:5])
    x_1 = tf.keras.activations.softmax(x[...,5:])
    new_x = tf.keras.layers.concatenate([x_0, x_1], axis = -1) # tf.concat()
    return new_x

#7-2:
##W = 'C:/Users/user/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def create_VGG(input_shape=(224, 224, 3), num_outs = 7):
    
    inputs = Input(shape = input_shape)
    vgg_model = VGG16(weights = 'imagenet', include_top = False, input_tensor = inputs)
    vgg_model.trainable = False # freeze
    ##for layer in vgg_model.layers: layer.trainable = False

                      
    # classification
    x = vgg_model.output
    x = Flatten()(x)
    x = Dense(64)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    
    x = Dense(32)(x)
    x = LeakyReLU()(x)
    
    outs  = Dense(num_outs, activation =custom_activations)(x)
    model = tf.keras.Model(inputs = inputs, outputs = outs)
    return model

model= create_VGG()
##model.summary()

#8: train the model
opt = RMSprop(learning_rate = 0.001)
##opt = Adam(learning_rate=0.0001)

model.compile(optimizer = opt, loss = custom_loss, metrics =  [IOU, custom_acc])
##model.compile(optimizer = opt, loss = 'mse', metrics =  [IOU])

ret = model.fit(x_train, y_train, epochs =100, batch_size =32, verbose = 0)

model.evaluate(x_train, y_train, verbose = 2)
model.evaluate(x_test,  y_test, verbose = 2)

#9: plot accuracy and loss
fig, ax = plt.subplots(1, 3, figsize = (10, 6))
ax[0].plot(ret.history['loss'], "r-")
ax[0].set_title('train loss')
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

ax[1].plot(ret.history['IOU'], "g-")
ax[1].set_title('IOU')
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('IOU')

ax[2].plot(ret.history['custom_acc'], "b-")
ax[2].set_title('custom_acc')
ax[2].set_xlabel('epochs')
ax[2].set_ylabel('custom_acc')
fig.tight_layout()
plt.show()

#10: predict k samples and display results
k = 16
train_pred = model.predict(x_train[:k])
test_pred  = model.predict(x_test[:k])

true_label = train_dataset["label"][:k]
pred_label = np.argmax(train_pred[:, -2:], axis=-1)
train_matches = np.sum(true_label== pred_label)
print("train_matches ={}/{}".format(train_matches, k))

true_label = test_dataset["label"][:k]
pred_label = np.argmax(test_pred[:, -2:], axis=-1)
test_matches = np.sum(true_label== pred_label)
print("test_matches ={}/{}".format(test_matches, k))

class_name=['Cat', 'Dog']
def create_label(pred): 
    p = pred[:, -2:] # [c1, c2] in [pc, x, y, w, h, c1, c2]
    p = tf.argmax(p, axis=-1)
    return p.numpy()

def display_images(img, pred, true_box=None, size= 224):

    box  = pred[:, 1:5]*size # [x, y, w, h] in [pc, x, y, w, h, c1, c2]

    label = create_label(pred)

    k =  pred.shape[0] # k = 16
    fig = plt.figure(figsize=(8, k//2))
   
    for i in range(k):
        plt.subplot(k//4, 4, i+1)
        plt.title(class_name[label[i]])

        a_img = (img[i]*255).astype('uint8')

        # predicted box
        x, y, w, h = box[i]
        w2 = w/2
        h2 = h/2
        xmin = int(x - w2)
        xmax = int(x + w2)
        ymin = int(y - w2)
        ymax = int(y + w2)                                
        cv2.rectangle(a_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

        if true_box is not None:
            xmin, ymin, xmax, ymax = true_box[i]
            cv2.rectangle(a_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            
        plt.imshow(a_img)
        plt.axis("off")
    fig.tight_layout()
    plt.show()

display_images(x_train[:k], train_pred, train_dataset["box"][:k])
display_images(x_test[:k],  test_pred)
