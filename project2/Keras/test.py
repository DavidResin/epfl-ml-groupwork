#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:49:44 2017

@author: Nicolas
"""
import os
import numpy as np
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Dropout
from sklearn.cross_validation import train_test_split
from keras.layers import Conv1D, MaxPooling1D
from helpers import load_data_and_labels, clean_files
from keras.models import Sequential
from keras.layers.core import Reshape, Flatten

MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
#
# first, build index mapping words in the embeddings set
# to their embedding vector

print("bonjour")
print("Indexing word vectors")
embeddings_index = {}
f = open('../glove/glove.twitter.27B.100d.txt', encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
#
print('Found %s word vectors.' % len(embeddings_index))

        
# Load data from processed files
if not os.path.exists('../processed/train_pos_CN.txt') \
    or not os.path.exists('../processed/train_neg_CNN.txt'):
        print('Cleaned files do not exist,'+' Cleaning the files')
        clean_files()

train,labels,test=load_data_and_labels('../processed/train_pos_CNN.txt','../processed/train_neg_CNN.txt','../processed/test_CNN.txt')
MAX_SEQUENCE_LENGTH = max(len(x) for x in train)
# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train)
sequences_train = tokenizer.texts_to_sequences(train)
sequences_test = tokenizer.texts_to_sequences(test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
##        
#        
xtrain = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
xtest = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

# prepare embedding matrix



# split data training and testing
X_train, X_test, y_train, y_test = train_test_split( xtrain, labels, test_size=0.2, random_state=42)

drop = 0.2

nb_epoch = 1
batch_size = 32
num_filters=128

model = Sequential()
model.add(Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))

model.add(Conv1D(num_filters, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(drop))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(X_train, y_train,
          epochs=5 ,
          batch_size=batch_size)
score = model.evaluate(X_test, y_test, batch_size=batch_size)