# 3_4_reviews.py
import random

import nltk
import collections


# nltk.download('movie_reviews')

# print(nltk.corpus.movie_reviews.fileids())
# print(nltk.corpus.movie_reviews.categories())
# print(nltk.corpus.movie_reviews.fileids('neg'))
# print(nltk.corpus.movie_reviews.raw())

# 퀴즈
# 영화 리뷰 데이터로부터
# 80%로 학습하고 20%에 대해 정확도를 구하세요
def make_vocab():
    fileids = nltk.corpus.movie_reviews.fileids()
    total = [nltk.corpus.movie_reviews.raw(name) for name in fileids]
    # print(total[1])
    # print(len(total))

    raw = ''.join(total)
    # print(len(raw))
    # print(len(nltk.corpus.movie_reviews.raw()))

    tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(raw)
    # print(len(tokens))

    stop_words = nltk.corpus.stopwords.raw('english')

    tokens = [w for w in tokens if w not in stop_words]
    tokens = [w for w in tokens if len(w) > 1]
    # print(len(tokens))

    # freq = nltk.FreqDist(tokens)
    freq = collections.Counter(tokens)
    most2000 = freq.most_common(2000)
    # print(most2000[:5])

    return [t[0] for t in most2000]


def make_features(words, vocab):
    features, uniques = {}, set(words)
    for v in vocab:
        features['has_{}'.format(v)] = (v in uniques)
    return features


vocab = make_vocab()
print(vocab[:5])

words = nltk.corpus.movie_reviews.words('neg/cv000_29416.txt')
print(words[:5])

raw = nltk.corpus.movie_reviews.raw('neg/cv000_29416.txt')
words = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(raw)
print(words)

# make_features(words, vocab)
neg = [nltk.corpus.movie_reviews.words(n) for n in nltk.corpus.movie_reviews.fileids('neg')]
pos = [nltk.corpus.movie_reviews.words(n) for n in nltk.corpus.movie_reviews.fileids('pos')]

random.shuffle(neg)
random.shuffle(pos)

neg_data = [(make_features(t, vocab), 'neg') for t in neg]
pos_data = [(make_features(t, vocab), 'pos') for t in pos]

train_set = neg_data[:800] + pos_data[:800]
test_set = neg_data[800:] + pos_data[800:]

clf = nltk.NaiveBayesClassifier.train(train_set)
acc = nltk.classify.accuracy(clf, test_set)
print(acc)
