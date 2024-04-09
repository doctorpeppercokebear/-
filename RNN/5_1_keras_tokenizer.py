import keras

# 5_1_keras_tokenizer.py
import keras


sent = "if you want to build a ship, don't drum up people to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea."

tokenizer = keras.preprocessing.text.Tokenizer(num_words=10)
tokenizer.fit_on_texts(sent.split())
print(tokenizer)
print(tokenizer.index_word)
print(tokenizer.word_index)
print()

print(tokenizer.index_word[9])      # build
print(tokenizer.index_word[10])     # a
print(tokenizer.index_word[11])     # ship
print()

print(tokenizer.texts_to_sequences(['build', 'a', 'ship']))
print(tokenizer.texts_to_sequences([['build'], ['a'], ['ship']]))
print(tokenizer.texts_to_sequences([['build', 'a', 'ship']]))
print()


seq = [[7, 8, 9, 1, 2], [13, 14, 5], [6, 7]]
print(tokenizer.sequences_to_texts(seq))
print()

print(keras.preprocessing.sequence.pad_sequences(seq))
print(keras.preprocessing.sequence.pad_sequences(seq, padding='pre'))
print(keras.preprocessing.sequence.pad_sequences(seq, padding='post'))
print()

print(keras.preprocessing.sequence.pad_sequences(seq))
print(keras.preprocessing.sequence.pad_sequences(seq, maxlen=7))
print(keras.preprocessing.sequence.pad_sequences(seq, maxlen=3))
print()

print(keras.preprocessing.sequence.pad_sequences(seq, maxlen=3, truncating='pre'))
print(keras.preprocessing.sequence.pad_sequences(seq, maxlen=3, truncating='post'))
