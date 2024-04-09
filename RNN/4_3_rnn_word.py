
'''
퀴즈
아래 문장에 대해 동작하는 RNN 모델을 만드세요.
'''

import keras
import numpy as np
from sklearn import preprocessing

sentence = "a rolling stone gathers no moss"

rnn = __import__('4_3_rnn_char_3')
rnn.rnn_char_3(sentence.split())
print('==')

