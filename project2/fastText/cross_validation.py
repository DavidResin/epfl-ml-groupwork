#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 09:43:26 2017

@author: Nicolas
"""
import numpy as np
from helpers import load_data_and_labels,clean_files
from sklearn.model_selection import KFold
import fasttext
import os


# Clean file if not exist
if not os.path.exists('../processed/train_pos_fastText_full.txt') \
    or not os.path.exists('../processed/train_neg_fastText_full.txt'):
        print('Cleaned fastText files do not exist')
        clean_files()
        
        
# Load data from processed files
train,labels,test=load_data_and_labels('../processed/train_pos_fastText_full.txt','../processed/train_neg_fastText_full.txt','../processed/test_data_fastText.txt')

# define the parameters for the fastText classifier
window=10
epochs=20

i=0
# create random indices of the rows size
num_row = len(labels)
indices = np.random.permutation(num_row)
# Define the number of fold for the cross-validation
fold=10;
k_fold = KFold(n_splits=fold)
accuracy=np.zeros((fold))
temp=k_fold.split(indices)
for train_indices, test_indices in k_fold.split(labels):
    # randomize the cross-val indices with the indices created above
    train_indices=indices[train_indices]
    test_indices=indices[test_indices]
    # Create the correct label in front of every tweets as : '__label__<X>'
    # For the training set
    with open('../processed/fastText_train_labels.txt', 'w', encoding="utf-8") as f:
        for indice in train_indices:
             f.write('__label__' +str(labels[indice]) +' '+train[indice]+ '\n')
    # for the testing set 
    with open('../processed/fastText_test_labels.txt', 'w', encoding="utf-8") as f:
        for indice in test_indices:
             f.write('__label__'  +str(labels[indice]) +' '+train[indice]+ '\n')
    
    # Build the fastText classifier 
    classifier = fasttext.supervised('../processed/fastText_train_labels.txt', 'model_cros_val', label_prefix='__label__', ws =window,epoch=epochs)
    # evaluate how the classifier performs on the testing set
    result = classifier.test('../processed/fastText_test_labels.txt')
    # Saving for every iterations the accuracy
    accuracy[i]=result.precision
    i=i+1
