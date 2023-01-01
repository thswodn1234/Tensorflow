import numpy as np
   
def MSE(y, t):
    return np.sum((y-t)**2)/t.size
    
x = np.arange(12) # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]
t = np.arange(12)

w = 0.5     # 초기값
b = 0
lr = 0.001  # 0.01, learning rate

loss_list = [ ]
for epoch in range(200):
    y = w*x + b                      # calculate the output   
    dW = np.sum((y-t)*x)/(2*x.size)  # gradients
    dB = np.sum((y-t))/(2*x.size)
    
    w = w - lr*dW     # update parameters
    b = b - lr*dB

    y = w*x + b       # calculate the output
    loss = MSE(y, t)
    loss_list.append(loss)
##    if not epoch%10:
##        print("epoch={}: w={:>8.4f}. b={:>8.4f}, loss={}".format(epoch, w, b, loss))

print("w={:>.4f}. b={:>.4f}, loss={:>.4f}".format(w, b, loss))

import matplotlib.pyplot as plt
plt.plot(loss_list)
plt.show() 
