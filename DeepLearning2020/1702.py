import tensorflow as tf
import numpy as np
#1
##y_true = np.array([0, 1, 2, 0, 1, 2])
##y_true = tf.keras.utils.to_categorical(y_true) # one-hot
y_true = np.array([[1, 0, 0], #0
                   [0, 1, 0], #1
                   [0, 0, 1], #2
                   [1, 0, 0], #0
                   [0, 1, 0], #1
                   [0, 0, 1]]);#2
               
y_pred = np.array([[0.3, 0.6, 0.1],  #1,
                   [0.6, 0.3, 0.1],  #0
                   [0.1, 0.3, 0.6],  #2
                   [0.3, 0.6, 0.1],  #1
                   [0.1, 0.6, 0.3],  #1
                   [0.3, 0.1, 0.6]]);#2
num_class = y_true.shape[1] # 3

#2: C and TOP_k
#2-1: threshold, and C in # 3-1, #4-1, and #6 in [step1701]
y_true1 = np.argmax(y_true, axis=1).flatten()
y_pred1 = np.argmax(np.cast['int'](y_pred>0.5), axis=1).flatten()
C = tf.math.confusion_matrix(y_true1, y_pred1)
print("y_true1=",y_true1) # y_true1= [0 1 2 0 1 2]
print("y_pred1=",y_pred1) # y_pred1= [1 0 2 1 1 2]
print("confusion_matrix(C)=", C)

#2-2: to find top-k index, in #3-2, #4-2
k=2
indx = tf.argsort(y_pred, axis=1, direction='DESCENDING')
TOP_k = indx[:,:k]
print("TOP_k = ", TOP_k)

#3
print("In each class, precision!")
#3-1: binary(1 above threshold=0.5, 0 below threshold= 0.5)  
for i in range(num_class):
    m = tf.keras.metrics.Precision(class_id = i)
    m.update_state(y_true, y_pred)
    tp = m.true_positives.numpy()
    fp = m.false_positives.numpy()
    p = m.result().numpy()
    print(" p_{} ={}, tp={}, fp= {}".format(i,p, tp, fp))
    
#3-2: the top-k classes with the highest predicted values
print("In each class, precision with top_k=", k)
for i in range(num_class):
    m = tf.keras.metrics.Precision(top_k=k, class_id = i)
    m.update_state(y_true, y_pred)
    tp = m.true_positives.numpy()
    fp = m.false_positives.numpy()
    p = m.result().numpy()
    print(" p_{} ={}, tp={}, fp= {}".format(i,p, tp, fp))
#4 
print("In each class, recall!")
#4-1: binary(1 above threshold=0.5, 0 below threshold= 0.5)
for i in range(num_class):
    m = tf.keras.metrics.Recall(class_id = i)
    m.update_state(y_true, y_pred)
    tp = m.true_positives.numpy()
    fn = m.false_negatives.numpy()
    r = m.result().numpy()
    print(" recall_{} ={}, tp={}, fn= {}".format(i,r, tp, fn))

#4-2: the top-k classes with the highest predicted values
print("In each class, recall with top_k=", k)
for i in range(num_class):
    m = tf.keras.metrics.Recall(top_k=k, class_id = i)
    m.update_state(y_true, y_pred)
    r = m.result().numpy()
    print(" recall_{} ={}, tp={}, fn= {}".format(i,r, tp, fn))
