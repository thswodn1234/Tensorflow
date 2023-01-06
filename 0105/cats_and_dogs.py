from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tf.keras.preprocessing import image
import os
tf.test.is_built_with_cuda()

# 네트워크 구성


model = models.Sequential()
# 입력 특성 맵에 적용 할 필터 수: 32, 윈도우 사이즈, 활성화함수, 입력 데이터 규격: 150*150, RGB 3채널
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))  # 최대 풀링 연산 적용할 윈도우 사이즈 - 다운샘플링(크기 축소)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))  # 윈도우 사이즈
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))  # 윈도우 사이즈
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))  # 윈도우 사이즈
# 여기까지 합성곱 기반 층(지역 패턴 추출 층)

# 여기서부터 완전 연결 층(전역 패턴 추출, 분류기)
model.add(layers.Flatten())  # 1차원 텐서(벡터)로 변환
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # 출력층: 최상위층, 분류 결과물 확률 꼴로 변환.

model.summary()


# 모델 컴파일


model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.adam_v2.Adam(learning_rate=0.001),
    metrics=['acc']
)

# 데이터 전처리


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './cats_and_dogs_small/train',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

valid_generator = test_datagen.flow_from_directory(
    './cats_and_dogs_small/test',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)


# 모델 훈련
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=1,
    validation_data=valid_generator,
    validation_steps=50
)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Tranining Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.suptitle('Accuracy & Loss')
plt.tight_layout()

plt.show()

# 데이터 증식

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.5,
    horizontal_flip=True,
    fill_mode='nearest'

)

# 데이터 증식 결과 시각화해서 살펴보기

fnames = sorted([os.path.join('./cats_and_dogs_small/train/cats', fname)
                for fname in os.listdir('./cats_and_dogs_small/train/cats')])
img_path = fnames[7]

img = image.load_img(img_path, target_size=(150, 150)
                     )  # 이미지 읽어오기, 크기 150 * 150으로 변환

x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

plt.figure(figsize=(5, 5))
i = 1
for batch in datagen.flow(x, batch_size=1):
    plt.subplot(2, 2, i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    plt.xticks([])
    plt.yticks([])
    i += 1
    if i == 5:
        break
plt.tight_layout()
plt.show()
