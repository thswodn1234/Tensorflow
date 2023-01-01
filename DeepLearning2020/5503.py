'''
ref1: http://www.robots.ox.ac.uk/~vgg/data/pets/
ref2: https://github.com/mpecha/Oxford-IIIT-Pet-Dataset
ref3: https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Localizer.ipynb?pli=1#scrollTo=LvKWKQ8QjCSx
'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image # pip install pillow
import os

import xml.etree.ElementTree as ET
import cv2  # pip install opencv_python

#1: extract Bounding Box from xml
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

#2: 
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
        image_name, class_id, species,  breed_id = line.split()
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
            train_dataset["label"].append(int(species)-1) # Cat: 0, Dog: 1
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
print("train_dataset['image'].shape=", train_dataset['image'].shape)# (3671, 224, 224, 3)
print("test_dataset['image'].shape=", test_dataset['image'].shape)  # (3678, 224, 224, 3)

#3: generate a batch 
def mini_batch(batch_size = 8):
    n = train_dataset["image"].shape[0]
    idx = np.random.choice(n, size=batch_size)
    
    image = train_dataset["image"][idx]
    box = train_dataset["box"][idx]
    label = train_dataset["label"][idx]
    name = train_dataset["name"][idx]
    return image, box, label, name
     
batch= mini_batch(8)

#4: display a batch
label_name = ['Cat', 'Dog']
def display_images(batch):
    img, box, label, name = batch
    fig = plt.figure(figsize=(8, 4))
    
    for i in range(img.shape[0]):
        plt.subplot(2, img.shape[0]//2, i+1)
        a_img = img[i].astype('uint8')
        xmin, ymin, xmax, ymax = box[i]
        cv2.rectangle(a_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
        plt.imshow(a_img)
        plt.title(name[i], fontsize=8)
        plt.axis("off")
        
    fig.tight_layout()
    plt.show()  
display_images(batch)
