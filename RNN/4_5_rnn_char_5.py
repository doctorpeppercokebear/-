# 4_4_rnn_char_4.py
import keras
import numpy as np
from sklearn import preprocessing


def make_xy(words):
    # 원 핫 백터를 구성하려면 모든 단어를 합쳐야 한다. tensorcoffelaptop
    bin = preprocessing.LabelBinarizer()        # 범주형 데이터를 이진 형태로 변환
    bin.fit(list(''.join(words)))

    max_len = max([len(w) for w in words])

    x, y = [], []
    for w in words:
        # 'tea' -> 'tea    ' 7글자 맞추기 위해 공백을 준다.
        w+= '_' * (max_len - len(w))

        onehot = bin.transform(list(w))
        x.append(onehot[:-1])
        y.append(np.argmax(onehot[1:], axis=1))

    return np.int32(x), np.int32(y), bin.classes_


def rnn_char_5(words):
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
    필요없는 부분을 삭제하세요
    '''
    valids = [len(w) -1 for w in words]
    print(*[vocab[pp[:vv]] for pp, vv in zip(p_arg, valids)], sep='\n')
    print(''.join(vocab[pp[:vv]]) for pp, vv in zip(p_arg, valids))


'''
퀴즈
길이가 다른 단어의 목록에 대해 동작하도록 수정하세요
'''

rnn_char_5(['tensor', 'tea', 'desktop'])   # 단어의 길이를 맞춰야함

