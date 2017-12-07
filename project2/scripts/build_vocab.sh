#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat ../twitter-datasets/train_pos_proc.txt ../twitter-datasets/train_neg_proc.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ../outputs/vocab.txt
