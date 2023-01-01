import tensorflow as tf
from tensorflow.keras.datasets import reuters
import numpy as np
import matplotlib.pyplot as plt
#1
(x_train, y_train), (x_test, y_test) = reuters.load_data()

#2
##(x_train, y_train), (x_test, y_test) = reuters.load_data(skip_top=15, num_words=101)
##print("x_train.shape=",x_train.shape) # (8982,)
##print("y_train.shape=",y_train.shape) # (8982,)
##print("x_test.shape=", x_test.shape)  # (2246,)
##print("y_test.shape=", y_test.shape)  # (2246,)

#3
##nlabel, count = np.unique(y_train, return_counts=True)
##print("nlabel:", nlabel)
##print("count:",  count)
##print("# of Class:",  len(nlabel) ) # 46 

##print("max(x_train words):", max(len(x) for x in x_train))# 2376
##print("max(x_test words):",  max(len(x) for x in x_test)) # 1032

#https://github.com/SteffenBauer/KerasTools/blob/master/KerasTools/datasets/decode.py
label = ('cocoa','grain','veg-oil','earn','acq','wheat','copper','housing',
              'money-supply','coffee','sugar','trade','reserves','ship','cotton',
              'carcass','crude','nat-gas','cpi','money-fx','interest','gnp',
              'meal-feed','alum','oilseed','gold','tin','strategic-metal',
              'livestock','retail','ipi','iron-steel','rubber','heat','jobs',
              'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead')
##print("x_train[0]:", x_train[0])

#4: decoding x_train[n], reverse from integers to words
# 0, 1, 2: 'padding', 'start of sequence', and 'unknown word'
n = 0 # n = 584, it's cocoa news
print("y_train[{}]={}".format(n, y_train[n]))
print("News label: {}".format(label[y_train[n]]))

index = reuters.get_word_index()
reverse_index  = dict([(value, key) for (key, value) in index.items()]) 
review = " ".join( [reverse_index.get(i-3, "?") for i in x_train[n]] )
print("review of x_train[{}]:\n{}".format(n, review))
