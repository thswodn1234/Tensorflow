import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#1
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

# one-hot encoding: 'mse', 'categorical_crossentropy'  
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
##print("y_train=", y_train)
##print("y_test=", y_test)

#3: change shapes for Conv1D
x_train= np.expand_dims(x_train, axis=2) # shape = (120, 4, 1)
x_test = np.expand_dims(x_test, axis=2)  # shape = ( 30, 4, 1)
print("x_train.shape:", x_train.shape)
print("x_test.shape:",  x_test.shape)

#4: build a model with Conv1D
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=10,
                                 kernel_size=4,
                                 input_shape=(4,1), activation='sigmoid'))
##model.add(tf.keras.layers.Dense(units=3, activation='softmax'))
model.add(tf.keras.layers.Conv1D(filters=3,
                                 kernel_size=1,
                                 activation='softmax'))
model.add(tf.keras.layers.Flatten())
model.summary()

#5: train and evaluate the model
opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

ret = model.fit(x_train, y_train, epochs=100, verbose=0) # batch_size = 32
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
