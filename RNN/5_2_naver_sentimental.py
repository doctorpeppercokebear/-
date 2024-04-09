# 5_2_naver_sentimental.py
import re
import keras
import matplotlib.pyplot as plt
import numpy as np

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    '''
    정규 표현식 두가지 표현 방식
    findall : 검색
    sub :  치환 , 대체
    '''

    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def make_xy(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    f.readline()

    x, y = [], []
    for line in f:
        user_id, doc, label = line.split('\t')
        # print(user_id, doc, label)

        x.append(clean_str(doc).split())
        y.append(int(label))

    f.close()

    return x[:10000], np.int32(y[:10000])           # x, y 데이터를 10000개로 제한 -> cpu로 학습해야 되기에 제한
    # return x, np.int32(y)



x_train_src, y_train = make_xy('data/ratings_train.txt')
x_test_src, y_test = make_xy('data/ratings_test.txt')
print(len(x_train_src), len(x_test_src))
print(*x_train_src[:3], sep='\n')

# 그래프
# lens = sorted([len(t) for t in x_train])
# plt.bar(range(len(lens)), lens)
# plt.show()

vocab_size = 2000
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts((x_train_src))

x_train_seq = tokenizer.texts_to_sequences(x_train_src)
x_test_seq = tokenizer.texts_to_sequences(x_test_src)

seq_length = 25
x_train_pad = keras.preprocessing.sequence.pad_sequences(x_train_seq, maxlen=seq_length)
x_test_pad = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=seq_length)

'''
1번
'''
# 15만개 데이터 이기에 학습 하기에 어렵다  -> 실전에서는 적은 데이터양이다.
# eye = np.eye(vocab_size, dtype=np.int32)
# x_train = eye[x_train_pad]
# x_test = eye[x_test_pad]
# print(x_train.shape, x_test.shape)

# x_train, x_test = x_train_pad, x_test_pad
#
# model = keras.Sequential([
#     keras.layers.Input(shape=x_train.shape[1:]),
#     keras.layers.LSTM(50),          # LSTM 모델 사용
#     keras.layers.Dense(1, activation='sigmoid')
# ])
# # model.summary()


'''
2번
'''
x_train, x_test = x_train_pad, x_test_pad
model = keras.Sequential([
    keras.layers.Input(shape=x_train.shape[1:]),    # 2차원 -> 원 핫 백터로 바뀌지 않았기에
    keras.layers.Embedding(vocab_size, 100),        # Embedding: 토큰을 숫자로 바꿔 준다, 유니크한 원 핫 백터로 바꿔준다., 2차원을 3차원으로 바꿔준다.
    keras.layers.LSTM(50, return_sequences=False),          # LSTM 모델 사용, 3차원 -> 2차원
    keras.layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.binary_crossentropy,
              metrics='acc')

# 모델 학습
model.fit(x_train, y_train, epochs=10, verbose=2,
          batch_size=128, validation_data=(x_test, y_test))

'''
퀴즈
아래 리뷰에 대해 긍정/부정을 판단하세요
'''
review = "이거 재밌다고 한 놈 찾아라. 너무 재밌다."

tokens_1 = clean_str(review).split()

for i in range(len(tokens_1)):
    part = tokens_1[:i+1]
    tokens_2 = tokenizer.texts_to_sequences(tokens_1)
    tokens_2 = keras.preprocessing.sequence.pad_sequences(tokens_2, maxlen=seq_length)

    p = model.predict(tokens_2, verbose=0)
    # print('{:.4f}'.format(p[0, 0]), [tokenizer.index_word for j in part])
    print('{:.4f}'.format(p[0, 0]), ''.join(part))
