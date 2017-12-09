#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat ../process/train_pos.txt ../process/train_neg.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ../process/vocab.txt
