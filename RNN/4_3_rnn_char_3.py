# 4_3_rnn_char_3.py
import keras
import numpy as np
from sklearn import preprocessing


# 퀴즈
# 앞의 코드를 모든 단어에 대해 동작하도록 수정하세요
def make_xy(word):
    bin = preprocessing.LabelBinarizer()
    onehot = bin.fit_transform(list(word))
    print(onehot)
    print(bin.classes_)

    x = onehot[:-1]
    y = np.argmax(onehot[1:], axis=1)

    # vocab = sorted(set(word))
    # vocab = np.array(vocab)     # ['e' 'n' 'o' 'r' 's' 't']

    # x[np.newaxis, :, :]
    # x[:, np.newaxis, :]
    # x[:, :, np.newaxis]
    return x[np.newaxis], y[np.newaxis], bin.classes_
    # return x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0]), bin.classes_
    # return x.reshape(1, *x.shape), y.reshape(1, -1), bin.classes_


def rnn_char_3(word):
    x, y, vocab = make_xy(word)
    print(x.shape, y.shape)     # (1, 5, 6) (1, 5)

    model = keras.Sequential([
        keras.layers.Input(shape=x.shape[1:]),
        keras.layers.SimpleRNN(16, return_sequences=True),
        keras.layers.Dense(x.shape[-1], activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=10, verbose=2)

    p = model.predict(x, verbose=0)
    p_arg = np.argmax(p, axis=2)
    print(p_arg)

    print('acc :', np.mean(p_arg == y))
    print(vocab[p_arg])


if __name__ == 'main':
    # rnn_char_3('tensor')
    # rnn_char_3('desktop')
    rnn_char_3('deep learning')

