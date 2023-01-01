import tensorflow as tf
import numpy as np

#1
y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([1, 0, 2, 1, 1, 2])

#2 
m = tf.keras.metrics.Accuracy()
m.update_state(y_true, y_pred)  #m.count = 3, m.total=6
print("accuracy from f.keras.metrics.Accuracy()=", m.result().numpy() )

#3
C = tf.math.confusion_matrix(y_true, y_pred)
print("confusion_matrix=", C)

correct = tf.linalg.diag_part(C)
col_sum = tf.reduce_sum(C, axis=0)
row_sum = tf.reduce_sum(C, axis=1)
total   = tf.reduce_sum(C)  #  len(y_true), len(y_pred)

accuracy    = tf.reduce_sum(correct)/total
precision_i = correct/col_sum
recall_i    = correct/row_sum
f1_i = 2*(precision_i*recall_i)/(precision_i+recall_i) # harmonic mean of precision and recall
f1_i = tf.where(tf.math.is_nan(f1_i), tf.zeros_like(f1_i), f1_i) # nan to 0.0
print("accuracy=",    accuracy.numpy())
print("precision_i=", precision_i.numpy())
print("recall_i=",    recall_i.numpy())
print("f1_i=",    f1_i.numpy())

#4:  micro, macro, weighted avg in precision, recall 
tp = tf.reduce_sum(correct)    # notice : correct pairs such as (0,0), (1,1), (2,2)
fp = tf.reduce_sum(col_sum - correct) # in this case, fp == fn
fn = tf.reduce_sum(row_sum - correct) 
precision = tp/(tp + fp)  
recall    = tp/(tp + fn)

count = tf.math.bincount(y_true) # support  in sklearn.metrics
print("count =", count)
print("precision(micro avg)=", precision.numpy())
print("precision(macro avg)=", tf.reduce_sum(precision_i)/precision_i.shape[0])
w=  tf.cast(count, dtype = tf.float64)/y_true.shape[0]  # tf.cast(total, dtype = tf.float64)
weightedAvgP = tf.reduce_sum(precision_i*w)
print("precision(weighted avg)=", weightedAvgP)

print("recall(micro avg)=", recall.numpy())
print("recall(macro avg)=", tf.reduce_sum(recall_i)/recall_i.shape[0])
weightedAvgR = tf.reduce_sum(recall_i*w)
print("recall(weighted avg)=", weightedAvgR)

#5: pip install sklearn
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import accuracy_score,precision_score, recall_score
print("--- sklearn.metrics ---")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

print("accuracy=", accuracy_score(y_true, y_pred)) # normalize=True
print("precision_i=", precision_score(y_true, y_pred, average=None))
print("precision(micro avg)=", precision_score(y_true, y_pred, average='micro'))
print("precision(macro avg)=", precision_score(y_true, y_pred, average='macro'))

print("recall_i=", recall_score(y_true, y_pred, average=None))
print("recall(micro avg)=", recall_score(y_true, y_pred, average='micro'))
print("recall(macro avg)=", recall_score(y_true, y_pred, average='macro'))
