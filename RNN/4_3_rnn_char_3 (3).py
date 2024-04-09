import keras
import numpy as np
from sklearn import preprocessing

# 퀴즈
# 앞의 코드를 모든 단어에 대해 동작하도록 수정하세요
def make_xy(word):
    # word는 문자열로 이루어진 단어입니다. 예를 들어 'tensor'나 'deep learning'과 같습니다.
    # 이 함수는 word를 one-hot 인코딩한 행렬로 변환하고, x와 y를 분리하여 반환합니다.
    # x는 word의 첫 글자부터 마지막 글자 전까지의 one-hot 인코딩 행렬입니다.
    # y는 word의 두 번째 글자부터 마지막 글자까지의 one-hot 인코딩 행렬에서 가장 큰 값의 인덱스를 담은 벡터입니다.
    # bin.classes_는 word에 포함된 고유한 문자들의 리스트입니다.
    bin = preprocessing.LabelBinarizer()
    onehot = bin.fit_transform(list(word)) # word를 one-hot 인코딩한 행렬을 만듭니다.
    print(onehot)

    x = onehot[:-1] # word의 첫 글자부터 마지막 글자 전까지의 행렬을 x에 저장합니다.
    y = np.argmax(onehot[1:], axis=1) # word의 두 번째 글자부터 마지막 글자까지의 행렬에서 가장 큰 값의 인덱스를 y에 저장합니다.

    # vocab = sorted(set(word))
    #
    # vocab = np.array(vocab)     # ['e' 'n' 'o' 'r' 's' 't']

    # return x[np.newaxis], y[np.newaxis], bin.classes_    # x, y, bin.classes_를 반환합니다. x와 y는 차원을 하나 늘려서 3차원 텐서로 만듭니다.
    return x.reshape(1, *x.shape), y.reshape(1, -1), bin.classes_



def rnn_char_3(word):
    # word는 문자열로 이루어진 단어입니다. 예를 들어 'tensor'나 'deep learning'과 같습니다.
    # 이 함수는 word를 입력으로 받아서 다음 문자를 예측하는 RNN 모델을 만들고 학습시킵니다.
    x, y, vocab = make_xy(word) # word를 x, y, vocab으로 변환합니다.
    print(x.shape, y.shape)     # x와 y의 모양을 출력합니다. (1, 5, 6) (1, 5)

    model = keras.Sequential([ # 순차적인 모델을 만듭니다.
        keras.layers.Input(shape=x.shape[1:]), # 입력층을 정의합니다. x의 모양에서 배치 차원을 제외한 나머지를 입력층의 모양으로 합니다. (5, 6)
        keras.layers.SimpleRNN(16, return_sequences=True), # 16개의 유닛을 가진 RNN 층을 추가합니다. return_sequences를 True로 하여 모든 타임스텝의 출력을 반환합니다.
        keras.layers.Dense(x.shape[-1], activation='softmax') # x의 마지막 차원과 같은 크기의 유닛을 가진 밀집층을 추가합니다. 활성화 함수로 softmax를 사용하여 각 문자의 확률을 출력합니다. (6)
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.1), # 옵티마이저로 Adam을 사용하고 학습률을 0.1로 설정합니다.
                  loss=keras.losses.sparse_categorical_crossentropy, # 손실 함수로 sparse_categorical_crossentropy를 사용합니다. y가 one-hot 인코딩이 아니라 정수 레이블이기 때문입니다.
                  metrics='acc') # 평가 지표로 정확도를 사용합니다.

    model.fit(x, y, epochs=10, verbose=2) # x와 y로 모델을 10번 학습시킵니다. 학습 과정을 간략하게 출력합니다.

    # 퀴즈
    # x, y를 3차원으로 변경했을 때의 에러를 수정하세요
    p = model.predict(x, verbose=0) # x에 대한 모델의 예측값을 p에 저장합니다. 예측 과정을 출력하지 않습니다.
    # print(p)

    p_arg = np.argmax(p, axis=2) # p에서 각 타임스텝마다 가장 큰 값의 인덱스를 p_arg에 저장합니다.
    # print(p_arg)                # [[0 1 4 2 3]]

    print('acc :', np.mean(p_arg == y)) # p_arg와 y가 일치하는 비율을 정확도로 출력합니다.

    # 퀴즈
    # 예측 결과를 디코딩하세요
    # print([i for i in p_arg[0]]) # p_arg의 첫 번째 원소를 리스트로 출력합니다. [0, 1, 4, 2, 3]
    # print([vocab[i] for i in p_arg[0]]) # p_arg의 첫 번째 원소에 해당하는 문자들을 리스트로 출력합니다. ['d', 'e', 'p', 'l', 'a']
    # print(vocab[p_arg[0]]) # p_arg의 첫 번째 원소에 해당하는 문자들을 배열로 출력합니다. ['d' 'e' 'p' 'l' 'a']
    print(vocab[p_arg]) # p_arg에 해당하는 문자들을 배열로 출력합니다. [['d' 'e' 'p' 'l' 'a']]

if __name__ =="main":
    # rnn_char_3('tensor')
    # rnn_char_3('desktop')
    rnn_char_3('deep learning') # 'deep learning'을 입력으로 rnn_char_3 함수를 실행합니다.
