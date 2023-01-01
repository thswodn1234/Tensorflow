import tensorflow as tf
#1
A = tf.constant([[1, 4,  1],
 	         [1, 6, -1],
		 [2, -1, 2]], dtype=tf.float32)
L_U, p = tf.linalg.lu(A)
print(L_U)
print(p)

#2: make P, L, U
U = tf.linalg.band_part(L_U, 0, -1) # Upper triangular
print(U)

L = tf.linalg.band_part(L_U, -1, 0) # Lower triangular
print(L)
L = tf.linalg.set_diag(L, [1, 1, 1]) # strictly lower triangular part of LU
print(L)

P = tf.gather(tf.eye(3), p)
print(P)

#3: check A= PLU 
#3-1:
print(tf.linalg.lu_reconstruct(L_U, p))

#3-2: calculate directly the same as #3-1
print(tf.matmul(P, tf.matmul(L, U))) # tf.gather(tf.matmul(L, U), p)

#4: solve AX = b using PLUx = b
b = tf.constant([[ 7],
                 [13],
                 [ 5]], dtype=tf.float32)
#4-1:
print(tf.linalg.lu_solve(L_U, p, b))


#4-2: calculate directly the same as #4-1
y = tf.linalg.triangular_solve(L, tf.matmul(tf.transpose(P), b))
print(y)

x = tf.linalg.triangular_solve(U, y, lower=False)
print(x)

#5: stuff: pivots, calulate det(A), rank(A)
D = tf.linalg.diag_part(L_U) # tf.linalg.diag_part(U)
print(D)

rank = tf.math.count_nonzero(D)
print(rank)

det_U = tf.reduce_prod(tf.linalg.diag_part(U)) # tf.linalg.det(U)
print(det_U)

det_L = tf.reduce_prod(tf.linalg.diag_part(L)) # # tf.linalg.det(L)  
print(det_L)

det_P = tf.linalg.det(P)
print(det_P)

det_A = det_P*det_L*det_U # tf.linalg.det(A)
print(det_A)
