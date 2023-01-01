import tensorflow as tf
import numpy as np

#1
y_true = np.array([[1, 0, 0], #0
                   [0, 1, 0], #1
                   [0, 0, 1], #2
                   [1, 0, 0], #0
                   [0, 1, 0], #1
                   [0, 0, 1]]);#2

# binary: 1 above threshold=0.5, 0 below threshold= 0.5                           
y_pred = np.array([[0.3, 0.6, 0.1], #1
                   [0.6, 0.3, 0.1], #0
                   [0.1, 0.3, 0.6], #2
                   [0.3, 0.6, 0.1], #1
                   [0.1, 0.6, 0.3], #1
                   [0.3, 0.1, 0.6]]);#2
#2
accuracy1 =tf.keras.metrics.binary_accuracy(y_true, y_pred)
print("accuracy1=", accuracy1)

#2-1
m= tf.keras.metrics.BinaryAccuracy()
m.update_state(y_true, y_pred)
# m.total = tf.reduce_sum(accuracy1)
# m.count = accuracy1.shape[0]
accuracy2 = m.result() # m.total/m.count
print("m.total={}, m.count={}".format(m.total.numpy(), m.count.numpy()))
print("accuracy2=", accuracy2.numpy())

#3: calculate confusion_matrix, C
y_true = y_true.flatten()
y_pred = np.cast['int'](y_pred.flatten()>0.5)

##y_true= tf.reshape(y_true, [y_true.shape[0]*y_true.shape[1]] )
##y_pred= tf.cast(y_pred>0.5, y_true.dtype)
##y_pred= tf.reshape(y_pred,  shape= y_true.shape )

##y_true= tf.keras.backend.flatten(y_true)
##y_pred= tf.cast(y_pred>0.5, tf.int32)
##y_pred= tf.keras.backend.flatten(y_pred)

print("y_true=",y_true)
print("y_pred=",y_pred)
C = tf.math.confusion_matrix(y_true, y_pred)
print("confusion_matrix(C)=", C)

#4:
m = tf.keras.metrics.Accuracy()
m.update_state(y_true, y_pred)
print("m.total={}, m.count={}".format(m.total.numpy(), m.count.numpy()))
accuracy3 = m.result()  # m.total/m.count
print("accuracy3=", accuracy3.numpy())

#5
#5-1
m = tf.keras.metrics.TruePositives()
m.update_state(y_true, y_pred)
tp = m.result()  # m.true_positives 
print("tp =", tp.numpy())

#5-2
m = tf.keras.metrics.TrueNegatives()
m.update_state(y_true, y_pred)
tn = m.result() # m.accumulator[0] 
print("tn=", tn.numpy())

#5-3
m = tf.keras.metrics.FalsePositives()
m.update_state(y_true, y_pred)
fp = m.result() # m.accumulator[0] sms
print("fp=", fp.numpy())

#5-4
m = tf.keras.metrics.FalseNegatives()
m.update_state(y_true, y_pred)
fn = m.result()# m.accumulator[0]  
print("fn=", fn.numpy())

accuracy4  = (tp + tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall    = tp/(tp+fn)
f1 = 2*tp/(2*tp + fp + fn) # harmonic mean of precision and recall
print("accuracy4 =", accuracy4.numpy())
print("precision =",precision.numpy())
print("recall =",   recall.numpy())
print("f1 score =", f1.numpy()) 
#6
#6-1
m = tf.keras.metrics.Precision()
m.update_state(y_true, y_pred)
print("m.true_positives=", m.true_positives.numpy())
print("m.false_positives", m.false_positives.numpy())
print("precision=", m.result().numpy())

#6-2
m = tf.keras.metrics.Recall()
m.update_state(y_true, y_pred)
print("m.true_positives=", m.true_positives.numpy())
print("m.false_negatives", m.false_negatives.numpy())
print("recall=", m.result().numpy())
