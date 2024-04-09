# 2_4_googlenet.py
import keras

# 퀴즈
# 구글넷 인셉션 v4 버전의 stem 부분의 아키텍쳐를 만드세요
inputs = keras.layers.Input(shape=(299, 299, 3))
outputs = keras.layers.Conv2D(32, 3, 2, 'valid', activation='relu')(inputs)
outputs = keras.layers.Conv2D(32, 3, 1, 'valid', activation='relu')(outputs)
outputs = keras.layers.Conv2D(64, 3, 1, 'same', activation='relu')(outputs)

out_left = keras.layers.MaxPool2D(3, 2, 'valid')(outputs)
out_rite = keras.layers.Conv2D(96, 3, 2, 'valid', activation='relu')(outputs)

outputs = keras.layers.concatenate([out_left, out_rite])

out_left = keras.layers.Conv2D(64, 1, 1, 'same', activation='relu')(outputs)
out_left = keras.layers.Conv2D(96, 3, 1, 'valid', activation='relu')(out_left)

out_rite = keras.layers.Conv2D(64, 1, 1, 'same', activation='relu')(outputs)
out_rite = keras.layers.Conv2D(64, (7, 1), 1, 'same', activation='relu')(out_rite)
out_rite = keras.layers.Conv2D(64, (1, 7), 1, 'same', activation='relu')(out_rite)
out_rite = keras.layers.Conv2D(96, 3, 1, 'valid', activation='relu')(out_rite)

outputs = keras.layers.concatenate([out_left, out_rite])

out_left = keras.layers.Conv2D(192, 3, 2, 'valid', activation='relu')(outputs)
out_rite = keras.layers.MaxPool2D(3, 2, 'valid')(outputs)

outputs = keras.layers.concatenate([out_left, out_rite])

model = keras.Model(inputs, outputs)
model.summary()
