#!/bin/bash
# download corpus
for corpus in nips kos
do 
    echo "downloading UCI $corpus corpus"
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.$corpus.txt
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.$corpus.txt.gz
    gunzip docword.$corpus.txt.gz
done