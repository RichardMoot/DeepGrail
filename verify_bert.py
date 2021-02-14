import numpy as np

treebank_sentences = 10
# treebank_sentences = 15748

all = ["sent%06d" %i for i in range(treebank_sentences)]

for ID in all:
    f = np.load('BERT_TLGbank/' + ID + '.npz')
    w = f['words']
    p1 = f['pos1']
    p2 = f['pos2']
    sup = f['super']
    e1 = f['bert']
    print(w)
    print(len(w), end=' ')
    print(len(p1), end=' ')
    print(len(p2), end=' ')
    print(len(sup), end=' ')
    print(e1.shape)
