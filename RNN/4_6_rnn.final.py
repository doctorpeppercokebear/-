# 4_6_rnn_final.py
import keras
import numpy as np
from sklearn import preprocessing

'''
퀴즈
긴 문장을 시퀀스 길이로 잘라서 단어 리스트를 만들고,
그에 맞게 동작하도록 나머지 코드를 수정하세요
'''

def make_xy(long_sentence, seq_length):
    bin = preprocessing.LabelBinarizer()
    bin.fit(list(long_sentence))

    x, y = [], []
    for i in range(len(long_sentence) - seq_length):
        w = long_sentence[i:i+seq_length+1]

        onehot = bin.transform(list(w))
        # print(onehot, end='\n\n')
        x.append(onehot[:-1])
        y.append(np.argmax(onehot[1:], axis=1))

    return np.int32(x), np.int32(y), bin.classes_


def rnn_final(long_sentence, seq_length):
    x, y, vocab = make_xy(long_sentence, seq_length)
    print(x.shape, y.shape)     #

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

    # print('acc :', np.mean(p_arg == y))

    print(long_sentence)
    print('*' + ''.join(vocab[p_arg[0]]), end='')
    for i in range(1, len(p_arg)):
        pp = p_arg[i]
        print(vocab[pp[-1]], end='')


long_sentence = "if you want to build a ship, don't drum up people to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea."
rnn_final(long_sentence, seq_length=20)



