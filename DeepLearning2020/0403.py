import tensorflow as tf
#1
a = tf.constant([1, 2, 3], dtype=tf.float32)
print(tf.norm(a)) # tf.linalg.norm(a)

#2
A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
print(tf.linalg.matrix_transpose(A))

#3
print(tf.linalg.det(A))

B =  tf.linalg.inv(A) 
print(B)
print(tf.matmul(A, B))  # tf.linalg.matmul(A, B)

