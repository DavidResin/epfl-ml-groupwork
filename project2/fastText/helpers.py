#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 23:50:42 2017

@author: Nicolas
"""
import numpy as np
from scipy.sparse import *
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

    with open('../processed/train_pos_fastText_full.txt', 'w', encoding="utf-8") as f:
        for sent in positive_string:
            f.write(sent + '\n')

    with open('../processed/train_neg_fastText_full.txt', 'w', encoding="utf-8") as f:
        for sent in negative_string:
            f.write(sent + '\n')

    with open('../processed/test_data_fastText.txt', 'w', encoding="utf-8") as f:
        for sent in test_string:
            f.write(sent + '\n')

def load_data_and_labels(positive_data_file, negative_data_file, test_data_file):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels for the training sets and split sentences for the testing set
    """
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
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [-1 for _ in negative_examples]
    labels = np.concatenate([positive_labels, negative_labels], 0)
    return [train, labels, test]

def submission(results):
    with open('sub.csv', 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        sub_writer = csv.DictWriter(csvfile, fieldnames)
        index = 0
        sub_writer.writeheader()
        for res in results:
            index=index+1
            sub_writer.writerow({'Id': str(index), 'Prediction': str(res[0][0])})
        print("Submission file created")

