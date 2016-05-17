import numpy as np
import lda
from scipy.io import loadmat
import scipy

# Read NIPS Dataset
with open('../data/docword.nips.txt', 'r') as df:
    num_docs = int(df.readline())
    num_words = int(df.readline())
    nnz = int(df.readline())

    X = scipy.sparse.lil_matrix((num_docs, num_words))

    for l in df:
        d, w, v = [int(x) for x in l.split()]
        X[d-1, w-1] = v

# read NIPS vocabulary
with open('../data/vocab.nips.txt', 'r') as vf:
    vocab = tuple(vf.read().split())

print("Vocabulary: {} words".format(len(vocab)))
print('Done reading NIPS dataset.')

# LDA to find topics
model = lda.LDA(n_topics=10, n_iter=1500, random_state=1)

print('Start fitting.')
model.fit(X.astype(int))  # model.fit_transform(X) is also available
print("Done fitting.")

topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
