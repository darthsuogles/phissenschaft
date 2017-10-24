from __future__ import print_function
import numpy as np
import string

from keras.models import Sequential
from keras import layers


# generate alphabet: http://stackoverflow.com/questions/16060899/alphabet-range-python
alphabet = string.ascii_lowercase
number_of_chars = len(alphabet)

# generate char sequences of length 'sequence_length'
# out of alphabet and store the next char as label (e.g. 'ab'->'c')
sequence_length = 10
sentences = [alphabet[i: i + sequence_length]
             for i in range(len(alphabet) - sequence_length)]
next_chars = [alphabet[i + sequence_length]
              for i in range(len(alphabet) - sequence_length)]

# Transform sequences and labels into 'one-hot' encoding
x = np.zeros((len(sentences), sequence_length, number_of_chars), dtype=np.bool)
y = np.zeros((len(sentences), number_of_chars), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, ord(char) - ord('a')] = 1
    y[i, ord(next_chars[i]) - ord('a')] = 1

# learn the alphabet with stacked LSTM
model = Sequential([
    layers.LSTM(16, return_sequences=True, input_shape=(sequence_length, number_of_chars)),
    layers.LSTM(16, return_sequences=False),
    layers.Dense(number_of_chars, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=60, verbose=1)

# prime the model with 'ab' sequence and let it generate the learned alphabet
sentence = alphabet[:sequence_length]
generated = sentence
for iteration in range(number_of_chars - sequence_length):
    x = np.zeros((1, sequence_length, number_of_chars))
    for t, char in enumerate(sentence):
        x[0, t, ord(char) - ord('a')] = 1.
    preds = model.predict(x, verbose=0)[0]
    next_char = chr(np.argmax(preds) + ord('a'))
    print(next_char, end='')
    generated += next_char
    sentence = sentence[1:] + next_char
