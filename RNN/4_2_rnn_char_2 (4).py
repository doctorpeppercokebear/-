# 4_2_rnn_char_2.py
import keras
import numpy as np



def rnn_char_2_sorted():
    # tensor -> enorst
    x = [[0, 0, 0, 0, 0, 1],  # t
         [1, 0, 0, 0, 0, 0],  # e
         [0, 1, 0, 0, 0, 0],  # n
         [0, 0, 0, 0, 1, 0],  # s
         [0, 0, 1, 0, 0, 0]]  # o
    y = [0, 1, 4, 2, 3]       # e n s o r

    model = keras.Sequential([
         keras.layers.Input(shape=(6,)),
         keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')


    model.fit(x, y, epochs=10, verbose=2)

    p = model.predict(x, verbose=0)
    print(p)

    p_arg = np.argmax(p, axis=1)
    print('acc :', np.mean(p_arg == y))


def rnn_char_2_simple():
    # tensor -> enorst
    x = [[0, 0, 0, 0, 0, 1],  # t
         [1, 0, 0, 0, 0, 0],  # e
         [0, 1, 0, 0, 0, 0],  # n
         [0, 0, 0, 0, 1, 0],  # s
         [0, 0, 1, 0, 0, 0]]  # o
    y = [0, 1, 4, 2, 3]       # e n s o r

    x = [x]
    y = [y]

    vocab = sorted('tensor')
    vocab = np.array(vocab)     # ['e' 'n' 'o' 'r' 's' 't']

    model = keras.Sequential([
        keras.layers.Input(shape=(5, 6)),
        keras.layers.SimpleRNN(16, return_sequences=True),
        keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=10, verbose=2)

    # 퀴즈
    # x, y를 3차원으로 변경했을 때의 에러를 수정하세요
    p = model.predict(x, verbose=0)
    print(p)

    p_arg = np.argmax(p, axis=2)
    print(p_arg)                # [[0 1 4 2 3]]

    print('acc :', np.mean(p_arg == y))

    # 퀴즈
    # 예측 결과를 디코딩하세요
    print([i for i in p_arg[0]])
    print([vocab[i] for i in p_arg[0]])

    print(vocab[p_arg[0]])
    print(vocab[p_arg])


# rnn_char_2_sorted()
rnn_char_2_simple()
