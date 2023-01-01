import tensorflow as tf
#1
A = tf.constant([[1, 4,  1],
	         [1, 6, -1],
		 [2, -1, 2]], dtype=tf.float32)
b = tf.constant([[ 7],
                 [13],
                 [ 5]], dtype=tf.float32)

#2
print(tf.linalg.det(A))

x = tf.matmul(tf.linalg.inv(A), b)
print(x)

#3
def all_close(x, y, tol=1e-5):
#	return tf.reduce_sum(tf.abs(x - y)) < tol
         return tf.reduce_sum(tf.square(x - y)) < tol
print(all_close(tf.matmul(A, x), b))

#4
x = tf.linalg.solve(A, b)
print(x)
print(all_close(tf.matmul(A, x), b))
