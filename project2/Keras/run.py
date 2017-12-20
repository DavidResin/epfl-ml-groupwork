#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:20:22 2017

@author: Nicolas
"""
import numpy as np
from helpers import load_data_and_labels, clean_files,submission,embedding_matrix
import os
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Merge, Convolution1D, Dropout,AveragePooling1D
from keras.layers.core import Flatten
from keras import optimizers

#difnine the path where is the glove twitter dataset
TWITTER_GLOVE_PATH='../glove/glove.twitter.27B.200d.txt'
nb_word = 20000
embedding_dim = 200




start=time.time()
# Clean file if not exist
if not os.path.exists('../processed/train_pos_CNN_full.txt') \
    or not os.path.exists('../processed/train_neg_CNN_full.txt'):
        print('Cleaned CNN files do not exist')
        clean_files()

# Load data from processed files
print('Load data from processed files')
train,labels,test=load_data_and_labels('../processed/train_pos_CNN_full.txt','../processed/train_neg_CNN_full.txt','../processed/test_data_CNN.txt')

sequence_length = max(len(x) for x in train)
# Vectorize the text samples into a 2D integer tensor with Tokenizer
tokenizer = Tokenizer(num_words=nb_word)
tokenizer.fit_on_texts(train)
sequences_train = tokenizer.texts_to_sequences(train)
sequences_test = tokenizer.texts_to_sequences(test)
# take only the index of words
word_index = tokenizer.word_index

# create the embedding matrix that will be the weight of our embedding layer
print('create embedding_matrix')
embedding_matrix_200 =embedding_matrix(TWITTER_GLOVE_PATH,word_index,nb_word,embedding_dim)

# put at the same lenght every sentences (lenght = max of all sentences)
xtrain = pad_sequences(sequences_train, maxlen=sequence_length)
Kaggle_sub = pad_sequences(sequences_test, maxlen=sequence_length)

# randomize data
num_row = len(labels)
indices = np.random.permutation(num_row)
train = xtrain[indices]
label_train=labels[indices]
# split data training, validation and testing if one wants
'''
nb_tweets_train=1000000
nb_tweets_test=100000
validation_split = int(0.20*nb_tweets_train)

x_train=X_train[: nb_tweets_train]
y_train=Y_train[: nb_tweets_train]
x_validation=X_train[nb_tweets_train+1 : nb_tweets_train+validation_split]
y_validation=Y_train[nb_tweets_train+1 : nb_tweets_train+validation_split]
x_test=X_train[nb_tweets_train+validation_split+1 :nb_tweets_train+validation_split+nb_tweets_test ]
y_test=Y_train[nb_tweets_train+validation_split+1 :nb_tweets_train+validation_split+nb_tweets_test]
'''
x_train=train
y_train=label_train

filters = [3,4,5]
num_filters = 120
drop = 0.6

nb_epoch = 3
batch_size = 30

convolutions = []
# this returns a tensor
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=nb_word + 1,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix_200],
                            input_length=sequence_length,
                            trainable=False)(inputs)
embedding2 = Embedding(input_dim=nb_word + 1,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix_200],
                            input_length=sequence_length,
                            trainable=True)(inputs)
for nb_filter in filters:
    conv = Convolution1D(num_filters, nb_filter, activation='relu')(embedding)
    maxpooling = AveragePooling1D(2)(conv)
    flatten=Flatten()(maxpooling)
    convolutions.append(flatten)
for nb_filter in filters:
    conv = Convolution1D(num_filters, nb_filter, activation='relu')(embedding2)
    maxpooling = AveragePooling1D(2)(conv)
    flatten=Flatten()(maxpooling)
    convolutions.append(flatten)

merged_tensor = Merge(mode='concat', concat_axis=1)(convolutions)
dense0=Dense(80,init='uniform', activation='relu')(merged_tensor)
dropout0 = Dropout(drop)(dense0)
dense1=Dense(50,init='uniform', activation='relu')(dropout0)
dropout1 = Dropout(drop)(dense1)
out = Dense(output_dim=2, init='uniform',activation='softmax')(dropout1)



model = Model(input=inputs, output=out)

Adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0005)

model.compile(loss='binary_crossentropy',
                                          optimizer=Adam,
                                          metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=nb_epoch ,
          batch_size=batch_size)

# score = model.evaluate(x_test, y_test, batch_size=batch_size)
# print(score[1])

result=model.predict(kaggle_sub, batch_size=None)

submission(result)
end = time.time()
run_time=end-start
print(run_time)
