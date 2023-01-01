import numpy as np
import matplotlib.pyplot as plt

def MSE(y, t):
    return np.sum((y-t)**2)/t.size

x = np.arange(12) # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]
t = np.arange(12)

w = 0.5
b = 0
lr = 0.001  # 0.01, learning rate
loss_list = [ ]

train_size = t.size # 12
batch_size = 4
K = train_size// batch_size # 3

for epoch in range(100):
    loss = 0
    for step in range(K):
        mask = np.random.choice(train_size, batch_size)
        x_batch = x[mask]
        t_batch = t[mask]
        
        y = w*x_batch + b                               # calculate the output
        dW = np.sum((y-t_batch)*x_batch)/(2*batch_size) # gradients
        dB = np.sum((y-t_batch))/(2*batch_size)
        
        w = w - lr*dW   # update parameters
        b = b - lr*dB
        
        y = w*x_batch + b       # calculate the output
        loss += MSE(y, t_batch) # calculate MSE
    loss /= K  # average loss
    loss_list.append(loss)
    if not epoch%10:
        print("epoch={}: w={:>8.4f}. b={:>8.4f}, loss={}".format(epoch, w, b, loss))

print("w={:>8.4f}. b={:>8.4f}, loss={}".format(w, b, loss))

plt.plot(loss_list)
plt.show() 
