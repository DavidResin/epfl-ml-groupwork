#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 09:43:26 2017

@author: Nicolas
"""
import numpy as np
from helpers import *
from sklearn.model_selection import KFold, cross_val_score
import fasttext
from helpers import *

train,labels,test=load_data_and_labels('../process/train_pos_full.txt','../process/train_neg_full.txt','../process/test_data.txt')
window=10
epochs=10
i=0
num_row = len(labels)
indices = np.random.permutation(num_row)
fold=10;
k_fold = KFold(n_splits=fold)
accuracy=np.zeros((fold))
temp=k_fold.split(indices)
for train_indices, test_indices in k_fold.split(labels):
    train_indices=indices[train_indices]
    test_indices=indices[test_indices]
    with open('../process/fastText_train_labels.txt', 'w', encoding="utf-8") as f:
        for indice in train_indices:
             f.write('__label__' +str(labels[indice]) +' '+train[indice]+ '\n')
    with open('../process/fastText_test_labels.txt', 'w', encoding="utf-8") as f:
        for indice in test_indices:
             f.write('__label__'  +str(labels[indice]) +' '+train[indice]+ '\n')
             
    classifier = fasttext.supervised('../process/fastText_train_labels.txt', '../fastText_classification/model', label_prefix='__label__', ws =window,epoch=epochs)
    
    result = classifier.test('../process/fastText_test_labels.txt')
    accuracy[i]=result.precision
    i=i+1
