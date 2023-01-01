'''
ref1: http://www.robots.ox.ac.uk/~vgg/data/pets/
ref2: https://github.com/mpecha/Oxford-IIIT-Pet-Dataset
Wrong trimap:  Egyptian_Mau_20.png:
'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image # pip install pillow
import os

#1: 
def load_oxford_pets_2(target_size= (224, 224), test_split_rate= 0.2):
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
            dataset["label"].append(int(species)-1) # Cat: species=1, Dog: species=2

            # read image and scale to target_size
            img = image.load_img(image_file, target_size=target_size)
            img = image.img_to_array(img)  # (224, 224, 3)
            dataset["image"].append(img)

            # read mask
            mask = image.load_img(mask_file, target_size= target_size, color_mode='grayscale')    
            mask = image.img_to_array(mask) # (224, 224, 1)
            dataset["mask"].append(mask)


    # change list to np.array
    dataset["name"]  = np.array(dataset["name"])
    dataset["label"] = np.array(dataset["label"])  # Cat: 0, Dog: 1
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

train_dataset, test_dataset = load_oxford_pets_2()
print("train_dataset['image'].shape=", train_dataset['image'].shape)# (5880, 224, 224, 3)
print("test_dataset['image'].shape=",  test_dataset['image'].shape) # (1469, 224, 224, 3)

#2: generate a batch
def mini_batch(batch_size = 8):
    n = train_dataset["image"].shape[0]
##    idx = np.random.randint(0, n, size=batch_size)
    idx = np.random.choice(n, size=batch_size)
    
    image = train_dataset["image"][idx]
    mask = train_dataset["mask"][idx]
    label = train_dataset["label"][idx]
    name = train_dataset["name"][idx]
    return image, mask, label, name
     
batch= mini_batch(4)

#3: display a batch
label_name = ['Cat', 'Dog']
def display_images(batch):
    img, mask,label, name = batch
    
    fig = plt.figure(figsize=(8, 4))
    n = img.shape[0]
    for i in range(n):
        plt.subplot(2, n, i+1)
        a_img = img[i].astype('uint8')
        plt.imshow(a_img)
        plt.title(name[i])
        plt.axis("off")
        
        plt.subplot(2, n, i+1+n)
        a_mask = mask[i][:,:,0]/255
        plt.imshow(a_mask)
        plt.title(label_name[label[i]]) # Cat/Dog
        plt.axis("off")
    fig.tight_layout()
    plt.show()
    
display_images(batch)
