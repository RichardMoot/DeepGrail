#!/Users/moot/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys, getopt
import os, os.path
import pickle
import operator

from keras.models import Model, load_model
from keras.layers import Bidirectional, Masking, Dense, Input, Dropout, LSTM, Activation, TimeDistributed, BatchNormalization, concatenate, Concatenate
from keras.layers.embeddings import Embedding
from keras.constraints import max_norm
from keras import regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.utils import to_categorical
from keras import backend as K
from sklearn.model_selection import train_test_split
from elmoformanylangs import Embedder

from grail_data_utils import *

np.random.seed(1)

def list_to_indices(list, item_to_index):
    ilist = []
    for i in range(len(list)):
        item = item_to_index[list[i]]
        ilist.append(item)
    return ilist
        
# load auxiliary mappings

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

super_to_index = load_obj('super_to_index')
index_to_super = load_obj('index_to_super')
pos1_to_index = load_obj('pos1_to_index')
index_to_pos1 = load_obj('index_to_pos1')
pos2_to_index = load_obj('pos2_to_index')
index_to_pos2 = load_obj('index_to_pos2')

numSuperClasses = len(index_to_super) + 1
numPos1Classes = len(index_to_pos1) + 1
numPos2Classes = len(index_to_pos2) + 1

# load corpus data
print('Loading corpus data')

# X, Y1, Y2, Z, vocabulary, vnorm, partsofspeech1, partsofspeech2, superset, maxLen = read_maxentdata('aa1.txt')
X, Y1, Y2, Z, vocabulary, vnorm, partsofspeech1, partsofspeech2, superset, maxLen = read_maxentdata('m2.txt')

print(np.shape(X))

# computing ELMo array

print('Loading ELMo embedder')

e = Embedder('/Users/moot/Software/FrenchELMo/')
cdir = "./TLGbank/"

for cursent in range(len(X)-1):

    fname = "sent%06d.npz" % cursent
    file = os.path.normpath(cdir + fname)

    words = X[cursent]
    pos1 = Y1[cursent]
    pos1 = list_to_indices(pos1, pos1_to_index)
    pos2 = Y2[cursent]
    pos2 = list_to_indices(pos2, pos2_to_index)
    super = Z[cursent]
    super = list_to_indices(super, super_to_index)
    embavg = e.sents2elmo([words], output_layer=-1)
    emb0 = e.sents2elmo([words], output_layer=0)
    emb1 = e.sents2elmo([words], output_layer=1)
    emb2 = e.sents2elmo([words], output_layer=2)

    np.savez(file, words=words, pos1=pos1, pos2=pos2, super=super, embavg=embavg, emb0=emb0, emb1=emb1, emb2=emb2)


