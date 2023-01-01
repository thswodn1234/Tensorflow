import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.Variable(2.0)
y = tf.Variable(3.0)

opt = tf.keras.optimizers.SGD(learning_rate=0.1) # learning_rate=0.001
##opt = tf.keras.optimizers.Adagrad(0.1) 
##opt = tf.keras.optimizers.Adam(0.1) 
##opt = tf.keras.optimizers.RMSprop(0.1)

loss_list = [ ]
for epoch in range(100):
        with tf.GradientTape() as tape:
                loss = x**2 + y**2               
        loss_list.append(loss.numpy())
        
##        grads = tape.gradient(loss, [x, y])
##        grads_list = [g for g in grads]
##        grads_and_vars = zip(grads_list, [x, y])
        
        dx, dy = tape.gradient(loss, [x, y])
        grads_and_vars = zip([dx, dy], [x, y])
        opt.apply_gradients(grads_and_vars)

##        if not epoch%10:
##                print("epoch={}: loss={}".format(epoch, loss.numpy()))
                
print ("x={:.5f}, y={:.5f}, loss={}".format(
        x.numpy(), y.numpy(), loss.numpy()))	
plt.plot(loss_list)
plt.show()
