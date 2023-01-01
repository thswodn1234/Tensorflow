import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#1
def createData(N=50):   
    np.random.seed(1)
    X0 = np.random.multivariate_normal(mean=[0.0, 0.0], cov= [[0.02, 0], [0, 0.01]], size=N)
    y_true0 = np.zeros(shape = (N,))

    X1 = np.random.multivariate_normal(mean=[0.0, 0.8], cov= [[0.01, 0], [0, 0.01]], size=N)
    y_true1 = np.ones(shape = (N,))

    X2 = np.random.multivariate_normal(mean=[0.3, 0.3], cov= [[0.01, 0], [0, 0.01]], size=N)
    y_true2 = np.ones(shape = (N,))*2

    X3 = np.random.multivariate_normal(mean=[0.8, 0.3], cov= [[0.01, 0], [0, 0.02]], size=N)
    y_true3 = np.ones(shape = (N,))*3

    X = np.vstack((X0, X1, X2, X3))
    y_true = np.hstack((y_true0, y_true1, y_true2, y_true3))
    return X, y_true

X, y_true = createData()   
y_true = tf.keras.utils.to_categorical(y_true) # 'mse', 'categorical_crossentropy'
##print("y_true=", y_true)

#2
n = 10  # number of neurons in a hidden layer
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=n, input_dim=2, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=4, activation='softmax'))
model.summary()

##model = tf.keras.Sequential()
##model.add(tf.keras.layers.Input(shape = (2,))) # shape = 2
##model.add(tf.keras.layers.Dense(units=n))
##model.add(tf.keras.layers.Activation('sigmoid'))
##model.add(tf.keras.layers.Dense(units=4))
##model.add(tf.keras.layers.Activation('softmax'))
##model.summary()


opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
##model.compile(optimizer=opt,loss='mse', metrics=['accuracy'])
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
##model.compile(optimizer=opt,
##              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

ret = model.fit(X, y_true, epochs=100, verbose=0) # batch_size=32
##print("len(model.layers):", len(model.layers))  # 2
##loss = ret.history['loss']
##plt.plot(loss)
##plt.xlabel('epochs')
##plt.ylabel('loss')
##plt.show()

#3
##print(model.get_weights())
##for i in range(len(model.layers)):
##    print("layer :", i, '-'*20)
##    w = model.layers[i].weights[0].numpy()
##    b = model.layers[i].bias.numpy()
##    print("weights[{}]: {}".format(i, np.array2string(w)))
##    print("bias[{}]:    {}".format(i, np.array2string(b)))

test_loss, test_acc = model.evaluate(X, y_true, verbose=2)

y_pred = model.predict(X)
##print("y_pred:", y_pred)
y_label = np.argmax(y_pred, axis = 1)
##print("y_label:", y_label)

C = tf.math.confusion_matrix(np.argmax(y_true, axis = 1), y_label)
print("confusion_matrix(C):", C)

#4: calculate the decision boundary
fig = plt.gcf()
fig.set_size_inches(6, 6)
plt.gca().set_aspect('equal')

markers = "ox+*"
colors  = "bgcm"
labels  = ("X0", "X1", "X2", "X3")
##label = y_true.flatten()          # loss='sparse_categorical_crossentropy'
label = np.argmax(y_true, axis = 1) # loss='mse', 'categorical_crossentropy'
for i, k in enumerate(np.unique(label)):
    plt.scatter(X[label==k, 0], X[label==k, 1],
                c = colors[i], marker=markers[i], label = labels[i])
plt.legend()

h = 0.01
x_min, x_max = X[:, 0].min()-h, X[:, 0].max()+h
y_min, y_max = X[:, 1].min()-h, X[:, 1].max()+h
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

sample = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(sample)
Z = np.argmax(Z, axis = 1)
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='red', linewidths=2)
plt.show()
