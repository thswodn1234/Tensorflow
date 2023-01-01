import tensorflow as tf
#1
a = tf.reshape(tf.range(12), shape = (3, 4))
print(a)

#2 
print(tf.reduce_min(a))
print(tf.reduce_min(a, axis= 0))
print(tf.reduce_min(a, axis= 1))

#3
print(tf.reduce_max(a))
print(tf.reduce_max(a, axis= 0))
print(tf.reduce_max(a, axis= 1))

#4
print(tf.reduce_sum(a))
print(tf.reduce_sum(a, axis= 0))
print(tf.reduce_sum(a, axis= 1))

#5
print(tf.reduce_mean(a))
print(tf.reduce_mean(a, axis= 0))
print(tf.reduce_mean(a, axis= 1))

#6
print(tf.reduce_prod(a))
print(tf.reduce_prod(a, axis= 0))
print(tf.reduce_prod(a, axis= 1))

#7
a = tf.reshape(tf.random.shuffle(tf.range(12)), shape = (3, 4))
print(a)

print(tf.argmin(a)) #  tf.argmin(a, axis =0)
print(tf.argmin(a, axis =1))
print(tf.argmax(a)) # tf.argmax(a, axis =0)
print(tf.argmax(a, axis =1))

#8
a = tf.random.shuffle(tf.range(12))
print(a)
print(tf.sort(a)) # direction="ASCENDING"
print(tf.sort(a, direction="DESCENDING"))

a = tf.reshape(a, shape=(3, 4))
print(a)
print(tf.sort(a)) # tf.sort(a, axis = 1)
print(tf.sort(a, axis = 0))
