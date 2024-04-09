# 4_4_rnn_char_4.py
import keras
import numpy as np
from sklearn import preprocessing


def make_xy(words):
    # 원 핫 백터를 구성하려면 모든 단어를 합쳐야 한다. tensorcoffelaptop
    bin = preprocessing.LabelBinarizer()        # 범주형 데이터를 이진 형태로 변환
    bin.fit_transform(''.join(words))

    x, y = [], []
    for w in words:
        onehot = bin.transform(list(w))
        x.append(onehot[:-1])
        y = np.argmax(onehot[1:], axis=1)

    return np.int32(x), np.int32(y), bin.classes_


def rnn_char_4(words):
    x, y, vocab = make_xy(words)
    print(x.shape, y.shape)

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

'''
퀴즈
길이가 같은 여러 개의 단어를 받는 버전으로 수정하세요
'''

rnn_char_4(['tensor', 'coffee', 'laptop'])

