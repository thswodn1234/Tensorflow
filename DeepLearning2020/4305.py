import tensorflow as tf
from tensorflow.keras.layers     import Input, Dense
import numpy as np
import matplotlib.pyplot as plt

#1: ref [step37_01], [그림 2.9]
##gpus = tf.config.experimental.list_physical_devices('GPU')
##tf.config.experimental.set_memory_growth(gpus[0], True)

#2
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype = np.float32)
y_and = np.array([[0],[0], [0],[1]], dtype = np.float32)    # AND
y_or = np.array([[0],[1],[1],[1]], dtype = np.float32)      # OR

#3: build a model
x_and = Input(shape = (2,))
out_and = Dense(units = 1, activation = 'sigmoid', name = 'and')(x_and)

x_or = Input(shape = (2, ))
out_or = Dense(units = 1, activation = 'sigmoid', name = 'or')(x_or)

model = tf.keras.Model(inputs = [x_and, x_or], outputs = [out_and, out_or])
model.summary()

#4: train and evaluate
opt = tf.keras.optimizers.RMSprop(learning_rate = 0.1)
model.compile(optimizer = opt, loss = 'mse', metrics = ['accuracy'])
ret = model.fit(x = [X, X], y = [y_and, y_or],
               epochs = 100, batch_size = 4, verbose = 0)    # silent
test = model.evaluate(x = [X, X], y = [y_and, y_or], verbose = 0)
print('total loss = ', test[0])        # test[1] + test[2]
print('AND: loss={}, acc={}'.format(test[1], test[3]))
print('OR: loss={}, acc={}'.format(test[2], test[4]))

#5: draw graph
plt.plot(ret.history['loss'],     'r--',  label='loss')
plt.plot(ret.history['and_loss'], 'g--', label='and_loss')
plt.plot(ret.history['or_loss'],  'b--', label='or_loss')
plt.plot(ret.history['and_accuracy'], 'g-', label='and_accuracy')
plt.plot(ret.history['or_accuracy'],  'b-', label='or_accuracy')
plt.xlabel('epochs')
plt.ylabel('loss and accuracy')
plt.legend(loc = 'best')
plt.show()
