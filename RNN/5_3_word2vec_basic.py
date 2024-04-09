import keras
import numpy
import nltk
import re

'''
퀴즈
주변 단어를 추출하는 함수를 만드세요.
'''

def extract_surrounds(token_count, center, window_size):
    start = max(center - window_size, 0)
    end = min(center + window_size + 1, token_count)

    return [i for i in range(start, end) if i != center]


def show_dataset(tokens, window_size, is_skipgram):
    token_count = len(tokens)
    for center in range(token_count):
        surrounds = extract_surrounds(token_count, center, window_size)
        # print(center, surrounds)

        if is_skipgram:
            print(*[(tokens[center], tokens[i]) for i in surrounds])
        else:
            print([tokens[i] for i in surrounds], tokens[center])






#            0    1    2    3   4     5    6   7     8
sentence = 'the quick brown fox jumps over the lazey dog'
tokens = sentence.split()

# show_dataset(tokens, window_size=2, is_skipgram=False)
show_dataset(tokens, window_size=2, is_skipgram=True)       #

# print(extract_surrounds(len(tokens), center=3, window_size=2))
# print(extract_surrounds(len(tokens), center=0, window_size=2))
# print(extract_surrounds(len(tokens), center=8, window_size=2))
