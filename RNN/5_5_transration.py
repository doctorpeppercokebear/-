# 5_5_translation.py
import numpy as np
import keras


def make_vocab_and_index(data):
    # 퀴즈
    # 영어와 한글을 합쳐서 단어장을 만드세요
    # vocab = sorted(set(''.join([t[0]+t[1] for t in data])))
    # vocab = ''.join(vocab) + 'SEP'

    eng = sorted(set(''.join([t for t, _ in data])))
    kor = sorted(set(''.join([t for _, t in data])))
    vocab = ''.join(eng + kor) + 'SEP'          # START, END, PAD
    print(vocab)

    char2idx = {c: i for i, c in enumerate(vocab)}
    # idx2char = {i: c for i, c in enumerate(vocab)}

    return vocab, char2idx


def make_xxy(data, char2idx):
    onehot = np.eye(len(char2idx), dtype=np.float32)

    x_enc, x_dec, y_dec = [], [], []
    for eng, kor in data:
        enc_in = [char2idx[c] for c in eng]
        dec_in = [char2idx[c] for c in 'S' + kor]
        target = [char2idx[c] for c in kor + 'E']

        x_enc.append(onehot[enc_in])
        x_dec.append(onehot[dec_in])
        y_dec.append(target)

    return np.float32(x_enc), np.float32(x_dec), np.float32(y_dec)


def show_translation(x_enc, x_dec, y_dec, vocab, char2idx):
    # 인코더
    enc_in = keras.layers.Input(shape=x_enc.shape[1:])
    _, enc_state = keras.layers.SimpleRNN(128, return_state=True)(enc_in)

    # 디코더
    dec_in = keras.layers.Input(shape=x_dec.shape[1:])
    output = keras.layers.SimpleRNN(128, return_sequences=True)(dec_in, initial_state=enc_state)
    output = keras.layers.Dense(len(vocab), activation='softmax')(output)

    model = keras.Model([enc_in, dec_in], output)
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit([x_enc, x_dec], y_dec, verbose=2, epochs=100)

    # 퀴즈
    # 'wind'와 'hero'를 한글로 번역하세요
    values = [('hero', '영웅'), ('wind', '바람')]
    x_enc, x_dec, _ = make_xxy(values, char2idx)

    p = model.predict([x_enc, x_dec], verbose=0)
    print(p.shape)

    p_arg = np.argmax(p, axis=2)
    print(p_arg)

    print([[vocab[i] for i in pp[:-1]] for pp in p_arg])


data = [('food', '음식'), ('hero', '영웅'),
        ('wind', '바람'), ('book', '도서'),
        ('head', '머리'), ('note', '공책')]

vocab, char2idx = make_vocab_and_index(data)
x_enc, x_dec, y_dec = make_xxy(data, char2idx)
show_translation(x_enc, x_dec, y_dec, vocab, char2idx)
