import numpy as np

def MSE(y, t):
    return np.sum((y-t)**2)/t.size

t = np.array([1,    2, 3,   4])    
y1 = np.array([0.5, 1, 1.5, 2])

print("MSE(t, y1)=", MSE(t, y1))

y2 = np.array([0.5, 1.5, 2.5, 3.5])
print("MSE(t, y2)=", MSE(t, y2))
