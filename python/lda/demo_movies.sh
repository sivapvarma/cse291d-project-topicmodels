#!/bin/bash
echo "starting robust spectral demo"
for corpus in movielens
do
    echo "use down_datasets.sh to download nips and nytimes datasets from UCI ML repo"
    # echo "downloading UCI $corpus corpus"
    # wget http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.$corpus.txt
    # wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.$corpus.txt
    # wget http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.$corpus.txt.gz
    # wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.$corpus.txt.gz
    # gunzip docword.$corpus.txt.gz
    
    echo "preprocessing, translate from docword.txt to scipy format"
    # python uci_to_scipy.py docword.$corpus.txt M_$corpus.full_docs.mat
    echo "preprocessing: removing rare words and stopwords"
    # python truncate_vocabulary.py M_$corpus.full_docs.mat vocab.$corpus.txt 50
    for loss in L2
    do
        for K in 10 15
        do
            echo "learning with nonnegative recover method using $loss loss..."
            python learn_topics.py M_$corpus.full_docs.mat settings.example vocab.$corpus.txt.trunc $K $loss demo_robust_$loss\_out.$corpus.$K
        done
    done
done
