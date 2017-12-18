#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:04:31 2017

@author: Nicolas
"""

import numpy as np
import csv
import re
#from gensim.models.keyedvectors import KeyedVectors
#from gensim.models import KeyedVectors

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
     #  Delete url and user 
    string = re.sub(r'<user>', ' ', string)
    string = re.sub(r'<url>', ' ', string)
    #  Replaced smily with meaning
    string = re.sub(r'\:\)', ' happy ', string)
    string = re.sub(r'\:\(', ' sad ', string)
    string = re.sub(r'\:\/', ' sarcasm ', string)
    string = re.sub(r'\<\d', ' love ', string)
    string = re.sub(r'&', ' and ', string)
    # Change the conjugaison
    string = re.sub(r"what's ", "what is ", string)
    string = re.sub(r" \'s ", " is ", string)
    string = re.sub(r" \'ve ", " have ", string)
    string = re.sub(r"can't ", "cannot ", string)
    string = re.sub(r"n't ", " not ", string)
    string = re.sub(r"i'm ", " i am ", string)
    string = re.sub(r"i've ", " i have ", string)
    string = re.sub(r"youre ", " you are ", string)
    string = re.sub(r"it's ", " it is ", string)
    string = re.sub(r"\'re ", " are ", string)
    string = re.sub(r"\'d ", " would ", string)
    string = re.sub(r"\'ll ", " will ", string)
    string = re.sub(r"don't ", " dont ", string)
    string = re.sub(r"im ", " i am ", string)

    # change the ponctuation 
    string = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", string)
    string = re.sub(r"\d", " ", string) 
    string = re.sub(r",", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\/", " ", string)
    string = re.sub(r"\^", " ^ ", string)
    string = re.sub(r"\+", " + ", string)
    string = re.sub(r"\-", " - ", string)
    string = re.sub(r"\=", " = ", string)
    string = re.sub(r"'", " ", string)
    string = re.sub(r"(\d+)(k)", r"\g<1>000", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r"\0s", "0", string)
    string = re.sub(r"e - mail", "email", string)
    string = re.sub(r"\s{2,}", " ", string)
   
    return string.strip().lower()

def clean_files():
    positive_examples = list(open('../twitter-datasets/train_pos_full.txt', "r", encoding="utf-8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open('../twitter-datasets/train_neg_full.txt', "r", encoding="utf-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    test_examples = list(open('../twitter-datasets/test_data.txt', "r", encoding="utf-8").readlines())
    test_examples = [s.strip() for s in test_examples]
    # process every words
    positive_string = [clean_str(sent) for sent in positive_examples]
    negative_string = [clean_str(sent) for sent in negative_examples]
    test_string = [clean_str(sent) for sent in test_examples]

    with open('../processed/train_pos_CNN_full.txt', 'w', encoding="utf-8") as f:
        for sent in positive_string:
            f.write(sent + '\n')

    with open('../processed/train_neg_CNN_full.txt', 'w', encoding="utf-8") as f:
        for sent in negative_string:
            f.write(sent + '\n')

    with open('../processed/test_data_CNN.txt', 'w', encoding="utf-8") as f:
        for sent in test_string:
            f.write(sent + '\n')


def load_data_and_labels(positive_data_file, negative_data_file, test_data_file):
    '''
    Load data take as input the pass of positive, negative and test  text file 
    and gives as output a merge of negative and positive and labels for CNN
    '''
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding="utf-8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding="utf-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    test = list(open(test_data_file, "r", encoding="utf-8").readlines())
    test = [s.strip() for s in test]
    # Split by words
    train = positive_examples + negative_examples
    # Generate labels
        # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    labels = np.concatenate([positive_labels, negative_labels], 0)
    labels = np.array(labels)
    return [train, labels, test]


def submission(results):
    with open('sub.csv', 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        sub_writer = csv.DictWriter(csvfile, fieldnames)
        index = 0
        sub_writer.writeheader()
        for res in results:
            index += 1
            if res[0]<res[1]:
                pred=1
            else:
                pred=-1
            sub_writer.writerow({'Id': str(index), 'Prediction': str(pred)})
        print("Submission file created")

def embedding_matrix(path_glove_twitter,word_index,nb_words,embedding_dim):
    '''
    Embedding_matrix take as input the path of the Glove librairy and index of 
    word constituating of the training set 
    And gives as output the corresponding representation of words 
    '''
    # create index mapping words in the embeddings  to their embedding vector
    embeddings_index = {}
    f = open(path_glove_twitter, "r", encoding="utf-8") 
    for line in f:
        values = line.split()
        word = values[0]
        # for each word we find the corresponding word vector
        embeddings_index[word] = np.asarray(values[1:], dtype='float32')
    f.close()

    # Create the embeding matrix corresponding to our Data-set
    embedding_matrix = np.zeros((nb_words + 1,embedding_dim))
    for word, i in word_index.items(): # Create the embeding matrix corresponding to our Dataset
        if i > nb_words: 
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

'''
def embedding_matrix_word2vec(path_glove_twitter,word_index,nb_words,embedding_dim):
    # first, build index mapping words in the embeddings set
    # to their embedding vector
    print("Indexing word vectors")
    model = KeyedVectors.load_word2vec_format('../glove/GoogleNews-vectors-negative300.bin', binary=True)  # C string format
    
    embedding_matrix = np.zeros((nb_words + 1,embedding_dim))
    for word, i in word_index.items():
        if i > nb_words:
            continue

        if word in model.vocab:
            embedding_matrix[i] = model.word_vec(word)
    return embedding_matrix
'''