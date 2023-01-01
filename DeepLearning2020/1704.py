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
          
y_pred = np.array([[0.3, 0.6, 0.1], #1
                   [0.6, 0.3, 0.1], #0
                   [0.1, 0.3, 0.6], #2
                   [0.3, 0.6, 0.1], #1
                   [0.1, 0.6, 0.3], #1
                   [0.3, 0.1, 0.6]]);#2

#2: using one-hot encoding in y_true
print("CategoricalAccuracy!")

#2-1
accuracy2_1= tf.keras.metrics.categorical_accuracy(y_true, y_pred)
print("accuracy2_1=", accuracy2_1.numpy())
#2-2
m = tf.keras.metrics.CategoricalAccuracy()
m.update_state(y_true, y_pred)
# m.total = tf.reduce_sum(accuracy2_1)
# m.count = accuracy2_1.shape[0]
accuracy2_2 = m.result() # m.total/m.count
print("m.total={}, m.count={}".format(m.total.numpy(),m.count.numpy()))
print("accuracy2_2=", accuracy2_2.numpy())

#2-3
top_k = 2 
accuracy2_3 = tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=top_k) 
print("top_k={}, accuracy2_3={}".format(top_k, accuracy2_3))

#2-4
m = tf.keras.metrics.TopKCategoricalAccuracy(k=top_k) # default k = 5
m.update_state(y_true, y_pred)
# m.total = tf.reduce_sum(accuracy2_3)
# m.count = accuracy2_3.shape[0]
accuracy2_4 = m.result()
print("m.total={}, m.count={}".format(m.total.numpy(),m.count.numpy()))
print("top_k={}, accuracy2_4={}".format(top_k, accuracy2_4.numpy()))

#3: using integer label in y_true
print("SparseCategoricalAccuracy!")
y_true = tf.argmax(y_true, axis = 1) # np.argmax(y_true, axis = 1)
y_true = tf.reshape(y_true, (-1,1))  # np.reshape(y_true, (-1, 1))
print("y_true=", y_true)

#3-1
accuracy3_1= tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
print("accuracy3_1=", accuracy3_1.numpy())
#3-2
m = tf.keras.metrics.SparseCategoricalAccuracy()
m.update_state(y_true, y_pred)
# m.total = tf.reduce_sum(accuracy3_1)
# m.count = accuracy3_1.shape[0]
accuracy3_2 = m.result() # m.total/m.count
print("m.total={}, m.count={}".format(m.total.numpy(),m.count.numpy()))
print("accuracy3_2=", accuracy3_2.numpy())

#3-3
top_k = 2 
accuracy3_3 = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=top_k) 
print("top_k={}, accuracy3_3={}".format(top_k, accuracy3_3))

#3-4
m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=top_k) # default k = 5
m.update_state(y_true, y_pred)
# m.total = tf.reduce_sum(accuracy3_3)
# m.count = accuracy3_3.shape[0]
accuracy3_4 = m.result()
print("m.total={}, m.count={}".format(m.total.numpy(),m.count.numpy()))
print("top_k={}, accuracy3_4={}".format(top_k, accuracy3_4.numpy()))
