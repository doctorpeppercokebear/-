# 2_5_alexnet.py
import keras

# 퀴즈
# slim에서 가져온 alexnet을 케라스로 변환하세요
inputs = keras.layers.Input(shape=(224, 224, 3))

outputs = keras.layers.Conv2D(64, 11, 4, 'valid', activation='relu', name='l1_conv')(inputs)
outputs = keras.layers.MaxPool2D(3, 2, 'same', name='l1_pool')(outputs)

outputs = keras.layers.Conv2D(192, 5, 1, 'same', activation='relu', name='l2_conv')(outputs)
outputs = keras.layers.MaxPool2D(3, 2, 'same', name='l2_pool')(outputs)

outputs = keras.layers.Conv2D(384, 3, 1, 'same', activation='relu', name='l3_conv1')(outputs)
outputs = keras.layers.Conv2D(384, 3, 1, 'same', activation='relu', name='l3_conv2')(outputs)
outputs = keras.layers.Conv2D(256, 3, 1, 'same', activation='relu', name='l3_conv3')(outputs)
outputs = keras.layers.MaxPool2D(3, 2, 'same', name='l3_pool')(outputs)

outputs = keras.layers.Conv2D(4096, 7, 1, 'valid', activation='relu', name='fc_dense1')(outputs)
outputs = keras.layers.Dropout(0.5)(outputs)
outputs = keras.layers.Conv2D(4096, 1, 1, 'same', activation='relu', name='fc_dense2')(outputs)
outputs = keras.layers.Dropout(0.5)(outputs)
outputs = keras.layers.Conv2D(1000, 1, 1, 'same', activation='softmax', name='fc_softmax')(outputs)

model = keras.Model(inputs, outputs)
model.summary()
