import tensorflow as tf
#1
A = tf.constant([[0, 1],
		 [1, 1],
		 [2, 1]], dtype=tf.float32)
b = tf.constant([[ 6],
                 [ 0],
                 [ 0]], dtype=tf.float32)
At = tf.transpose(A)
C = tf.matmul(At, A)
print(C)

#2
x = tf.linalg.solve(C, tf.matmul(At, b))
print(x)

#3
x2 = tf.matmul(tf.matmul(tf.linalg.inv(C), At), b)
print(x2)

#4
L_U, p = tf.linalg.lu(C)
x3 = tf.linalg.lu_solve(L_U, p, tf.matmul(At, b))
print(x3)

#4 
x4 = tf.linalg.lstsq(A, b)
print(x4)

# draw the line
m, c = x.numpy()[:,0]
import matplotlib.pyplot as plt
plt.gca().set_aspect('equal')
plt.scatter(x = A.numpy()[:,0], y = b.numpy())
t = tf.linspace(-1.0, 3.0, num=51)
b1 = m*t + c
plt.plot(t, b1, "b-")
plt.axis([-1, 10, -1, 10])
plt.show()
