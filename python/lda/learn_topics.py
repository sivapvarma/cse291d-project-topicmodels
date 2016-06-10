import numpy as np
import lda
from scipy.io import loadmat
import scipy
from time import time
import sys



#parse input args
if len(sys.argv) > 6:
    infile = sys.argv[1]
    settings_file = sys.argv[2]
    vocab_file = sys.argv[3]
    K = int(sys.argv[4])
    loss = sys.argv[5]
    outfile = sys.argv[6]

else:
    print("usage: ./learn_topics.py word_doc_matrix settings_file vocab_file K loss output_filename")
    print("for more info see readme.txt")
    sys.exit()

# load word-document matrix
M = scipy.io.loadmat(infile)['M']
# change to transpose between anchor and lda
X = M.T

# read in vocabulary
vocab = []
with open(vocab_file) as vfile:
    for line in vfile:
        vocab.append(line.strip())

print("Vocabulary: {} words".format(len(vocab)))
print('Done reading the dataset.')



model = lda.LDA(n_topics=K, n_iter=1500, random_state=1)

print('Start fitting.')
d = time()
model.fit(X.astype(int))  # model.fit_transform(X) is also available
d = time() - d
print("Done fitting in {:.1f} minutes.".format(d/60))

topic_word = model.topic_word_  # model.components_ also works
n_top_words = 10


# display
f = open(outfile+".topwords", 'w')
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i+1, ' '.join(topic_words)))
    print('Topic {}: {}'.format(i+1, ' '.join(topic_words)), file=f)
f.close()