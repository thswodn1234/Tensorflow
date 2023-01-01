import tensorflow as tf
import numpy as np
#1: 
def pad1d_infor(steps, kernel_size=2, strides=1,
                  dilation_rate=1, padding= 'valid'):   
    k = (kernel_size-1)*dilation_rate + 1
    if padding == 'valid':
        new_steps = int(np.ceil((steps - k + 1) / strides))
        pad_left, pad_right=(0, 0) 
       
    else: # 'same', 'casual'
        new_steps = int(np.ceil(steps/strides))     
        pad_width = max((new_steps  - 1) * strides + k - steps, 0)

        if padding == 'same':   
            pad_left  = pad_width//2
            pad_right = pad_width - pad_left
        if padding =='casual':               
            pad_left  = pad_width
            pad_right = 0
    return k, new_steps, (pad_left, pad_right)

#2: crate a 1D input data
A = np.array([1, 2, 3, 4, 5]).astype('float32')
length = A.shape[0] #len(len), 5
         
#3: padding in MaxPool1D [step37_02]
#3-1: 
new_k, new_steps, pads= pad1d_infor(steps=length, kernel_size=2,
                                      strides=2, padding= 'same')
print("new_k ={}, new_steps={}, pads={}".format(new_k,new_steps,pads))  
B1 = tf.pad(A, paddings=np.array([pads]))
print("B1=", B1.numpy())

#3-2: 
new_k, new_steps, pads= pad1d_infor(steps=length, kernel_size=4,
                                      strides=3, padding= 'same')
print("new_k ={}, new_steps={}, pads={}".format(new_k,new_steps,pads))  
B2 = tf.pad(A, paddings=np.array([pads]))
print("B2=", B2.numpy())

#4: padding in Conv1D
#4-1: 
new_k, new_steps, pads= pad1d_infor(steps=length, kernel_size=3,
                                      padding= 'same')                                
print("new_k ={}, new_steps={}, pads={}".format(new_k,new_steps,pads))  
B3 = tf.pad(A, paddings=np.array([pads]))
print("B3=", B3.numpy())

#4-2: 
new_k, new_steps, pads= pad1d_infor(steps=length, kernel_size=3,
                                      strides=2, padding= 'same')
print("new_k ={}, new_steps={}, pads={}".format(new_k,new_steps,pads))  
B4 = tf.pad(A, paddings=np.array([pads]))
print("B4=", B4.numpy())

#4-3: 
new_k, new_steps, pads= pad1d_infor(steps=length, kernel_size=3,
                                      dilation_rate=1, padding= 'casual')
print("new_k ={}, new_steps={}, pads={}".format(new_k,new_steps,pads))  
B5 = tf.pad(A, paddings=np.array([pads]))
print("B5=", B5.numpy())

#4-4: 
new_k, new_steps, pads= pad1d_infor(steps=length, kernel_size=3,
                                      dilation_rate=2, padding= 'same')
print("new_k ={}, new_steps={}, pads={}".format(new_k,new_steps,pads))  
B6 = tf.pad(A, paddings=np.array([pads]))
print("B6=", B6.numpy())

#4-5: 
new_k, new_steps, pads= pad1d_infor(steps=length, kernel_size=3,
                                      dilation_rate=2, padding= 'casual')
print("new_k ={}, new_steps={}, pads={}".format(new_k,new_steps,pads))  
B7 = tf.pad(A, paddings=np.array([pads]))
print("B7=", B7.numpy())

#4-6: 
new_k, new_steps, pads= pad1d_infor(steps=length, kernel_size=3,
                                      dilation_rate=3, padding= 'casual')
print("new_k ={}, new_steps={}, pads={}".format(new_k,new_steps,pads))  
B8 = tf.pad(A, paddings=np.array([pads]))
print("B8=", B8.numpy())
