import tensorflow as tf
#1
a = tf.constant([1, 2])
print(a + 1) # tf.add(a, 1), tf.math.add(a, 1)
print(a - 1) # tf.subtract(a, 1), tf.math.subtract(a, 1)
print(a * 2) # tf.multiply(a, 2), tf.math.multiply(a, 2)
print(a / 2) # tf.divide(a, 2), tf.math.divide(a, 2)

#2
b = tf.constant([3, 4])
print(a + b) # tf.add(a, b)
print(a - b) # tf.subtract(a, b)
print(a * b) # tf.multiply(a, b)
print(a / b)  # tf.divide(a, b)

#3
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([1, 2])
print(a + b) # tf.add(a, b)
print(a - b) # tf.subtract(a, b)
print(a * b) # tf.multiply(a, b)
print(a / b) # tf.divide(a, b)
