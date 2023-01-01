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

inputs = tf.keras.layers.Input(shape=(n+1,))
outputs = tf.keras.layers.Dense(units=1, use_bias=False)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
model.compile(optimizer=opt, loss='mse')
ret = model.fit(X, y_true, epochs=100, verbose=2)

#모델 동결(Freezing)
#ref1: https://github.com/leimao/Frozen_Graph_TensorFlow/blob/master/TensorFlow_v2/test.py
#ref2: https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/

#1: 모델을 하나의 시스니쳐를 갖는 ConcreteFunction으로 변환
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

#2: 동결함수 생성
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
frozen_func = convert_variables_to_constants_v2(full_model)

#3: 동결 그래프(frozen graph) 저장
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./RES",
                      name="frozen_graph.pb",
                      as_text=False)

#4: 모델구조 화면출력
##print(frozen_func.graph.as_graph_def())
##
##layers = [op.name for op in frozen_func.graph.get_operations()]
##print("-"* 20)
##print("model layers: ")
##for layer in layers:
##     print(layer)
##
##print("-" * 20)
##print("model inputs: ")
##print(frozen_func.inputs)
##print("model outputs: ")
##print(frozen_func.outputs)
