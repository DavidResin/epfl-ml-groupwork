#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 23:50:42 2017

@author: Nicolas
"""
import numpy as np
from scipy.sparse import *
import csv

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
            index += 1
            sub_writer.writerow({'Id': str(index), 'Prediction': str(res)})
        print("Submission file created")

