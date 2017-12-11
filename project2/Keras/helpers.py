#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:04:31 2017

@author: Nicolas
"""

import numpy as np
import csv
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r'\d', ' ', string)
    string = re.sub(r'\:\)', ' positive ', string)
    string = re.sub(r'\:\(', ' negative ', string)
    string = re.sub(r'[^A-Za-z0-9(),!?\'\`]', ' ', string)
    string = re.sub(r'\'s', ' \'s', string)
    string = re.sub(r'\'ve', " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", "  ", string)
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"<user>", " ", string)
    string = re.sub(r"<url>", " ", string)
    return string.strip().lower()

def clean_files():
    positive_examples = list(open('../twitter-datasets/train_pos_full.txt', "r", encoding="utf-8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open('../twitter-datasets/train_neg_full.txt', "r", encoding="utf-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    test_examples = list(open('../twitter-datasets/test_data.txt', "r", encoding="utf-8").readlines())
    test_examples = [s.strip() for s in test_examples]
    # Split by words
    positive_text = [clean_str(sent) for sent in positive_examples]
    negative_text = [clean_str(sent) for sent in negative_examples]
    test_text = [clean_str(sent) for sent in test_examples]

    with open('../processed/train_pos_CNN.txt', 'w', encoding="utf-8") as f:
        for sent in positive_text:
            f.write(sent + '\n')

    with open('../processed/train_neg_CNN.txt', 'w', encoding="utf-8") as f:
        for sent in negative_text:
            f.write(sent + '\n')

    with open('../processed/test_CNN.txt', 'w', encoding="utf-8") as f:
        for sent in test_text:
            f.write(sent + '\n')


def prediction():
    i=0
    f_pos = open('../processed/prediction.txt', 'r', encoding="utf-8")
    lines = f_pos.readlines()
    l=len(lines)
    results=np.zeros(l,dtype=int)
    for line in lines:
        results[i]=line.replace('__label__','')    
        i=i+1
    return results

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
            sub_writer.writerow({'Id': str(index), 'Prediction': str(res)})
        print("Submission file created")

