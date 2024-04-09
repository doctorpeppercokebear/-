# 2_1_functional.py
import keras
import numpy as np


# 퀴즈
# 아래 데이터에 대해 동작하는 모델을 만드세요
def functional_1():
    data = [[0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1]]
    data = np.int32(data)

    x, y = data[:, :-1], data[:, -1:]

    # model = keras.Sequential()
    # model.add(keras.layers.Input(shape=(2,)))
    # model.add(keras.layers.Dense(4, activation='relu'))
    # model.add(keras.layers.Dense(1, activation='sigmoid'))

    # 1번
    # ii = keras.layers.Input(shape=(2,))
    # d1 = keras.layers.Dense(4, activation='relu')
    # d2 = keras.layers.Dense(1, activation='sigmoid')
    #
    # o1 = d1.__call__(ii)
    # o2 = d2.__call__(o1)
    #
    # model = keras.Model(ii, o2)

    # 2번
    # ii = keras.layers.Input(shape=(2,))
    # o1 = keras.layers.Dense(4, activation='relu').__call__(ii)
    # o2 = keras.layers.Dense(1, activation='sigmoid').__call__(o1)
    #
    # model = keras.Model(ii, o2)

    # 3번
    # ii = keras.layers.Input(shape=(2,))
    # o1 = keras.layers.Dense(4, activation='relu')(ii)
    # o2 = keras.layers.Dense(1, activation='sigmoid')(o1)
    #
    # model = keras.Model(ii, o2)

    # 4번
    ii = keras.layers.Input(shape=(2,))
    out = keras.layers.Dense(4, activation='relu')(ii)
    out = keras.layers.Dense(1, activation='sigmoid')(out)

    model = keras.Model(ii, out)

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=10, verbose=2)


def functional_2():
    data = [[0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1]]
    data = np.int32(data)

    x1, x2, y = data[:, :1], data[:, 1:2], data[:, -1:]

    ii1 = keras.layers.Input(shape=(1,))
    out1 = keras.layers.Dense(4, activation='relu')(ii1)
    out1 = keras.layers.Dense(1, activation='relu')(out1)

    ii2 = keras.layers.Input(shape=(1,))
    out2 = keras.layers.Dense(4, activation='relu')(ii2)
    out2 = keras.layers.Dense(1, activation='relu')(out2)

    out = keras.layers.concatenate([out1, out2])
    out = keras.layers.Dense(1, activation='sigmoid')(out)

    model = keras.Model([ii1, ii2], out)

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    model.fit([x1, x2], y, epochs=10, verbose=2)


# 퀴즈
# AND와 XOR 데이터를 한번에 학습해서 각각의 결과를 도출하세요
def functional_3():
    data = [[0, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 0]]
    data = np.int32(data)

    x1, x2, y1, y2 = data[:, :1], data[:, 1:2], data[:, 2:3], data[:, 3:]

    ii1 = keras.layers.Input(shape=(1,))
    out1 = keras.layers.Dense(4, activation='relu')(ii1)
    out1 = keras.layers.Dense(1, activation='relu')(out1)

    ii2 = keras.layers.Input(shape=(1,))
    out2 = keras.layers.Dense(4, activation='relu')(ii2)
    out2 = keras.layers.Dense(1, activation='relu')(out2)

    out = keras.layers.concatenate([out1, out2])

    out1 = keras.layers.Dense(4, activation='relu')(out)
    out1 = keras.layers.Dense(1, activation='sigmoid')(out1)

    out2 = keras.layers.Dense(4, activation='relu')(out)
    out2 = keras.layers.Dense(1, activation='sigmoid')(out2)

    model = keras.Model([ii1, ii2], [out1, out2])

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    model.fit([x1, x2], [y1, y2], epochs=100, verbose=2)
    p1, p2 = model.predict([x1, x2], verbose=0)
    print(p1)
    print(p2)

    print(model.evaluate([x1, x2], [y1, y2], verbose=0))


# functional_1()
# functional_2()
functional_3()

