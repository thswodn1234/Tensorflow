# dataset download: c:> python 2300.py
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100, boston_housing, reuters, imdb
data  = boston_housing.load_data()
data  = imdb.load_data()   
index = imdb.get_word_index()
data  = reuters.load_data()
index = reuters.get_word_index()
data = mnist.load_data()
data = fashion_mnist.load_data()
data = cifar10.load_data()
data = cifar100.load_data()
