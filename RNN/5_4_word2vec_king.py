import keras
import numpy as np
import matplotlib.pyplot as plt

def make_vocab_and_vectors(corpus):
    '''
    퀴즈
    corpus로 부터 불용어가 제거된 2차원 토큰 리스트를 만드세요
    [['king', 'strong', 'man']]
    '''

    # corpus 가 먼저 tokens로 나눠져야 한다.
    tokens = [t.split() for t in corpus]
    print(tokens)

    # 불용어 목록
    stop_words = ['is', 'a', 'will', 'be']      # 제거할 문자열 리스트 변수에 지정
    tokens = [[w for w in t if w not in stop_words] for t in tokens]
    print(tokens)
    '''
    [['king', 'strong', 'man'],
     ['queen', 'wise', 'woman'],
     ['boy', 'young', 'man'], 
     ['girl', 'young', 'woman'], ....
    '''

    '''
    퀴즈
    tokens 로부터 단어장을 만들고
    해당 단어장을 사용해서 각가의 단어를 숫자로 변환하세요
    vocab 필요
    '''

    # vocab 만들기
    vocab = [w for t in tokens for w in t]          # 1차원으로 변경
    vocab = sorted(set(vocab))      # 유니크한 거 제거, sorted 로 정렬
    print(vocab)

    vectors = [[vocab.index(w) for w in t]for t in tokens]
    print(vectors)

    return vocab, vectors
def extract_surrounds(tokens, center, window_size):
    start = max(center - window_size, 0)
    end = min(center + window_size + 1, len(tokens))

    return [tokens[i] for i in range(start, end) if i != center]

def make_xy(vectors, vocab, window_size, is_skipgram):
    xx, yy = [], []
    for tokens in vectors:
        for center in range(len(tokens)):
            surrounds = extract_surrounds(tokens, center, window_size)
            # print(center, surrounds)

            if is_skipgram:
                # 퀴즈
                # 아래 코드를 xx, yy에 데이터를 넣는 코드로 수정하세요
                # print(*[(tokens[center], i) for i in surrounds])
                for s in surrounds:
                    xx.append(tokens[center])
                    yy.append(s)
            else:
                # print([i for i in surrounds], tokens[center])
                xx.append(surrounds)
                yy.append(tokens[center])

    print(xx)
    print(yy)

    x = np.zeros([len(xx), len(vocab)], dtype=np.float32)
    # print(x)

    for i, p in enumerate(xx):
        # print(i, p)
        if is_skipgram:
            x[i, p] = 1
        else:
            # print(p)
            onehot = [[k for k in range(len(vocab))] for t in p]
            # print(onehot)
            x[i] = np.mean(onehot, axis=0)
            # exit()
    print(x)
    return x, np.int32(yy)


class EpochCall(keras.callbacks.Callback):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if epoch % self.step == 0:
            print('{:05f} : {:.4f} {:.4f}'.format(epoch, logs['loss'], logs['acc']))




def show_model(x, y, vocab):
    model = keras.Sequential([
        keras.layers.Input(shape=(len(vocab),)),
        #(?, 2) = (?, 12) @ (12, 2)
        keras.layers.Dense(2, activation='relu'),
        # (?, 2) = (?, 12) @ (2, 12)
        keras.layers.Dense(len(vocab), activation='softmax'),
    ])
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')
    model.fit(x, y, verbose=0, epochs=20000, callbacks=[EpochCall(step=500)])

    layer1 = model.get_layer(index=0)
    print(layer1)

    w, b = layer1.get_weights()
    print(w.shape, b.shape)     #(12,2 )

    ii = vocab.index('king')
    print(w[ii])

    '''
    그래프
    '''
    for token, (x1, x2) in zip(vocab, w):
        plt.plot(x1, x2, 'ro')
        plt.text(x1, x2, token)

    plt.show()


corpus = [
    'king is a strong man',
    'queen is a wise woman',
    'boy is a young man',
    'girl is a young woman',
    'prince is a young king',
    'princess is a young queen',
    'man is strong',
    'woman is pretty',
    'price is a boy will be king',
    'princess is a will be queen',
]

vocab, vectors = make_vocab_and_vectors(corpus)


x, y = make_xy(vectors, vocab, window_size=2, is_skipgram=True)

show_model(x, y, vocab)
