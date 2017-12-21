#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 22:26:47 2017

@author: Nicolas
"""

import time
from helpers import load_data_and_labels,clean_files,submission
import fasttext
import os

# start computing time 
start=time.time()

# Clean file if not exist
if not os.path.exists('../processed/train_pos_fastText_full.txt') \
    or not os.path.exists('../processed/train_neg_fastText_full.txt'):
        print('Cleaned fastText files do not exist')
        clean_files()

# Load data from processed files
train,labels,test=load_data_and_labels('../processed/train_pos_CNN_full.txt','../processed/train_neg_CNN_full.txt','../processed/test_data_CNN.txt')

# Create the correct label in front of every tweets as : '__label__<X>'
with open('../processed/fastText_full_labels.txt', 'w', encoding="utf-8") as f:
    for sent, label in zip(train ,labels):
        f.write('__label__' +str(label) +' '+sent+ '\n')

# define the parameters for the fastText classifier
window=10
epochs=10

# Build the fastText classifier 
classifier = fasttext.supervised('../processed/fastText_full_labels.txt', 'model', label_prefix='__label__', ws=window, epoch=epochs)

# Create a summission csv file of the results
prob = classifier.predict_proba(test)
submission(prob)

# Compute the computing time
end = time.time()
run_time=end-start
print(run_time)