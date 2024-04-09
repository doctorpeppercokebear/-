# 2_2_cnn.py
import keras


# 퀴즈
# mnist 데이터셋에 대해 동작하는 CNN 모델을 만드세요
def lenet5():
    mnist = keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = mnist
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    x_train = x_train / 255
    x_test = x_test / 255

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),  # x_train.shape[1:]

        # keras.layers.Conv2D(filters=6,
        #                     kernel_size=(5, 5),
        #                     strides=(1, 1),
        #                     padding='SAME',
        #                     activation='relu'),
        # keras.layers.MaxPool2D(pool_size=(2, 2),
        #                        strides=(2, 2),
        #                        padding='SAME')

        keras.layers.Conv2D(6, 5, 1, 'SAME', activation='relu'),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Conv2D(16, 5, 1, 'VALID', activation='relu'),
        keras.layers.MaxPool2D(2, 2),

        keras.layers.Reshape([5 * 5 * 16]),
        # keras.layers.Flatten(),

        keras.layers.Dense(120, activation='relu'),
        keras.layers.Dense(84, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])
    model.summary()

    model.compile(optimizer=keras.optimizers.RMSprop(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2,
              validation_split=0.8)
    print('acc :', model.evaluate(x_test, y_test, verbose=0))


# 퀴즈
# mnist 데이터셋에 대해 99.5%가 넘는 모델을 만드세요
def my_net():
    mnist = keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = mnist

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    x_train = x_train / 255
    x_test = x_test / 255

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),

        keras.layers.Conv2D(16, 3, 1, 'same', activation='relu'),
        keras.layers.Conv2D(16, 3, 1, 'same', activation='relu'),
        keras.layers.MaxPool2D(2, 2),

        keras.layers.Conv2D(32, 3, 1, 'same', activation='relu'),
        keras.layers.Conv2D(32, 3, 1, 'same', activation='relu'),
        keras.layers.MaxPool2D(2, 2),

        keras.layers.Flatten(),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2,
              validation_split=0.2)
    print('acc :', model.evaluate(x_test, y_test, verbose=0))


# lenet5()
my_net()

# 5x5
# * * * * *
# * * * * *
# * * * * *
# * * * * *
# * * * * *

# 5x5 = 1

# 3x3 = 3x3
#       3x3 = 1

# 1x5 = 5x1
#       5x1 = 1











