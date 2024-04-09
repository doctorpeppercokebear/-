# 2_3_vgg16.py
import keras

# 퀴즈
# VGG16 모델의 아키텍처를 구성하세요
model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Conv2D(64, 3, 1, 'same', activation='relu'),
    keras.layers.Conv2D(64, 3, 1, 'same', activation='relu'),
    keras.layers.MaxPool2D(2, 2),

    keras.layers.Conv2D(128, 3, 1, 'same', activation='relu'),
    keras.layers.Conv2D(128, 3, 1, 'same', activation='relu'),
    keras.layers.MaxPool2D(2, 2),

    keras.layers.Conv2D(256, 3, 1, 'same', activation='relu'),
    keras.layers.Conv2D(256, 3, 1, 'same', activation='relu'),
    keras.layers.Conv2D(256, 3, 1, 'same', activation='relu'),
    keras.layers.MaxPool2D(2, 2),

    keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'),
    keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'),
    keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'),
    keras.layers.MaxPool2D(2, 2),

    keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'),
    keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'),
    keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'),
    keras.layers.MaxPool2D(2, 2),

    keras.layers.Flatten(),

    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dense(1000, activation='softmax'),
])
model.summary()
