import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def dataset(train_size=100): # tensorflow    
     tf.random.set_seed(1)
     x = tf.linspace(-5.0, 5.0, num=train_size)
     y = 3.0*x**3 + 2.0*x**2 + x + 4.0
     y += tf.random.normal([train_size], mean=0.0, stddev = 30.0)
     return x, y
x, y_true = dataset(20)

# n-차 다항식 회귀
n = 3
X = np.ones(shape = (len(x), n+1), dtype=np.float32)
for i in range(1, n+1):
     X[:, i] = x**i

##inputs = tf.keras.layers.Input(shape=(n+1,))
##outputs = tf.keras.layers.Dense(units=1, use_bias=False)(inputs)
##model = tf.keras.Model(inputs=inputs, outputs=outputs)
##model.summary()

#1: 모델 구조 로드
import json
file = open("./RES/1401.model", 'r')
json_model = json.load(file)
file.close()
model = tf.keras.models.model_from_json(json_model)
model.summary()

#2
opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
model.compile(optimizer=opt, loss='mse')

#3
latest = tf.train.latest_checkpoint("./RES/ckpt")
print('latest=', latest)
model.load_weights(latest) # 가중치 로드 
loss = model.evaluate(X, y_true, verbose=0) # 0 = silent
print("loss:", loss)
print("len(model.layers):", len(model.layers)) # 2
#print(model.get_weights())  # weights
print("weights:", model.layers[1].weights[0].numpy())

#4
y_pred = model.predict(X)
plt.scatter(x, y_true) 
plt.plot(x, y_pred, color='red')
plt.show()
