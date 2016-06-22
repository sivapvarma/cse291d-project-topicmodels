import numpy as np
import pickle

vocab_pkl_fname = 'newsgroups_vocab_1000.pkl'
vocab = pickle.load(open(vocab_pkl_fname, 'rb'))

afile = 'newsgroups_50.A'
A = np.loadtxt(afile)

print(A.shape)

outfile = 'newsgroups_40'
#display
f = open(outfile+".topwords", 'w')
for k in range(K):
    topwords = np.argsort(A[:, k])[-params.top_words:][::-1]
    print(vocab[anchors[k]], ':', end=' ')
    print(vocab[anchors[k]], ':', end=' ', file=f)
    for w in topwords:
        print(vocab[w], end=' ')
        print(vocab[w], end=' ', file=f)
    print("")
    print("", file=f)
f.close()
