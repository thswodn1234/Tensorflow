import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt

#1: 
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)

#2
def load_Iris(shuffle=False):   
    label={'setosa':0, 'versicolor':1, 'virginica':2}
    data = np.loadtxt("./Data/iris.csv", skiprows=1, delimiter=',',
                      converters={4: lambda name: label[name.decode()]})
    if shuffle:
        np.random.shuffle(data)
    return data

def train_test_data_set(iris_data, test_rate=0.2): # train: 0.8, test: 0.2
    n = int(iris_data.shape[0]*(1-test_rate))
    x_train = iris_data[:n,:-1]
    y_train = iris_data[:n, -1]
    
    x_test = iris_data[n:,:-1]
    y_test = iris_data[n:,-1]
    return (x_train, y_train), (x_test, y_test)
    
iris_data = load_Iris(shuffle=True)
(x_train, y_train), (x_test, y_test) = train_test_data_set(iris_data, test_rate=0.2)
##print("x_train.shape:", x_train.shape) # shape = (120, 4)
##print("x_test.shape:",  x_test.shape)  # shape = ( 30, 4)

# one-hot encoding: 'categorical_crossentropy'  
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# change shapes for Conv1D
x_train= np.expand_dims(x_train, axis=2) # shape = (120, 4, 1)
x_test = np.expand_dims(x_test, axis=2)  # shape = ( 30, 4, 1)

#3: build a functional cnn model
def create_cnn1d(input_shape, num_class = 3):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=10, kernel_size=4,activation='sigmoid')(inputs)
    x=  Dense(units=num_class, activation='softmax')(x)                           
##    x= Conv1D(filters=num_class,kernel_size=1,activation='softmax')(x)
    outputs =Flatten()(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
model = create_cnn1d(input_shape = (4,1))
model.summary()

#4: train the model
opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
ret = model.fit(x_train, y_train, epochs=100, verbose=0)

#5: evaluate the model
y_pred = model.predict(x_train)
y_label = np.argmax(y_pred, axis = 1)
C = tf.math.confusion_matrix(np.argmax(y_train, axis = 1), y_label)
print("confusion_matrix(C):", C)

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
