import tensorflow as tf
import numpy as np
#1 
def pad2d_infor(input_shape, kernel_size=(2,2), strides=(1,1),
                    dilation_rate=(1,1), padding= 'valid'):
    rows, cols = input_shape
    kH = (kernel_size[0]-1)*dilation_rate[0] + 1
    kW = (kernel_size[1]-1)*dilation_rate[1] + 1

    if padding == 'valid':
        new_rows = int(np.ceil((input_shape[0]-kH+1)/strides[0]))
        new_cols = int(np.ceil((input_shape[1]-kW+1)/strides[1]))
        pad_left, pad_right, pad_top, pad_bottom=(0, 0, 0, 0) 
       
    else: # 'same'
        new_rows = int(np.ceil(input_shape[0]/strides[0]))
        new_cols = int(np.ceil(input_shape[1]/strides[1]))
        
        pad_height = max((new_rows-1)*strides[0] + kH - input_shape[0], 0)       
        pad_width  = max((new_cols-1)*strides[1] + kW - input_shape[1], 0)

        pad_top    = pad_height//2
        pad_bottom = pad_height - pad_top    
        pad_left  = pad_width//2
        pad_right = pad_width - pad_left
    return (kH,kW),(new_rows,new_cols),[[pad_left,pad_right],[pad_top,pad_bottom]]

#2: crate a 2D input data
A = np.array([[1, 2, 3, 4, 5],
              [4, 3, 2, 1, 0],
              [5, 6, 7, 8, 9],
              [4, 3, 2, 1, 0],
              [0, 1, 2, 3, 4]],dtype='float32')
         
#3: padding in 2D
#3-1: 
new_k, new_shape, pads= pad2d_infor(input_shape=A.shape,
                                      kernel_size=(2,2), strides=(2,2), padding= 'valid')
print("new_k ={}, new_shape={}, pads={}".format(new_k, new_shape, pads))  
B1 = tf.pad(A, paddings=np.array(pads))
print("B1=", B1.numpy())

#3-2:
new_k, new_shape, pads= pad2d_infor(input_shape=A.shape,
                         kernel_size=(2,2), strides=(2,2), padding= 'same')
print("new_k ={}, new_shape={}, pads={}".format(new_k, new_shape, pads))  
B2 = tf.pad(A, paddings=np.array(pads))
print("B2=", B2.numpy())
