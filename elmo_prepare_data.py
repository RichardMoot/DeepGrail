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

maxLen = 266
numSuperClasses = len(index_to_super) + 1
numPos1Classes = len(index_to_pos1) + 1
numPos2Classes = len(index_to_pos2) + 1

# load corpus data
print('Loading training data')

X, Y1, Y2, Z, vocabulary, vnorm, partsofspeech1, partsofspeech2, superset, maxLen = read_maxentdata('m2.txt')

# computed ELMo array

e = Embedder('/Users/moot/Software/FrenchELMo/')
Xemb = e.sents2elmo(X)

del e

ll = len(Xemb)-1
Xarr= np.zeros((ll,maxLen,1024))\n
for i in range(ll):
    sl,t = (Xemb[i].shape)
    for j in range(sl):
        Xarr[i][j]= Xemb[i][j]

del Xemb

Y_super_indices = lists_to_indices(Z, super_to_index, maxLen)
Y_pos1_indices = lists_to_indices(Y1, pos1_to_index, maxLen)
Y_pos2_indices = lists_to_indices(Y2, pos2_to_index, maxLen)

# split the training data into the standard 60% train, 20% dev, 20% test 
X_train, X_testdev, X_train_embedding, X_testdev_embedding,\
Y_pos1_train_indices, Y_pos1_testdev_indices, Y_pos2_train_indices, Y_pos2_testdev_indices,\
Y_super_train_indices, Y_super_testdev_indices = train_test_split(X[0:-1], Xarr, Y_pos1_indices[0:-1],\
                                                               Y_pos2_indices[0:-1], Y_super_indices[0:-1], test_size=0.4)
X_test, X_dev, X_test_embedding, X_dev_embedding,\
Y_pos1_test_indices, Y_pos1_dev_indices, Y_pos2_test_indices, Y_pos2_dev_indices,\
Y_super_test_indices, Y_super_dev_indices = train_test_split(X_testdev, X_testdev_embedding, Y_pos1_testdev_indices,\
                                                               Y_pos2_testdev_indices, Y_super_testdev_indices, test_size=0.5)

# delete arrays which are no longer needed
del X
del Y1
del Y2
del Z
del vocabulary
del vnorm
del Xarr
del files

del Y_super_indices
del Y_pos1_indices
del Y_pos2_indices

Y_pos1_test_oh = to_categorical(Y_pos1_test_indices, num_classes = numPos1Classes)
Y_pos1_train_oh = to_categorical(Y_pos1_train_indices, num_classes = numPos1Classes)

Y_pos2_test_oh = to_categorical(Y_pos2_test_indices, num_classes = numPos2Classes)
Y_pos2_train_oh = to_categorical(Y_pos2_train_indices, num_classes = numPos2Classes)

Y_super_test_oh = to_categorical(Y_super_test_indices, num_classes = numSuperClasses)
Y_super_train_oh = to_categorical(Y_super_train_indices, num_classes=numSuperClasses)



Y_pos1_dev_oh = to_categorical(Y_pos1_dev_indices, num_classes = numPos1Classes)
Y_pos2_dev_oh = to_categorical(Y_pos2_dev_indices, num_classes = numPos2Classes)
Y_super_dev_oh = to_categorical(Y_super_dev_indices, num_classes = numSuperClasses)

# delete arrays which are no longer needed
del X_testdev
del X_testdev_embedding
del Y_pos1_testdev_indices
del Y_pos2_testdev_indices
del Y_super_testdev_indices


del Y_pos1_test_indices
del Y_pos1_train_indices
del Y_pos2_test_indices
del Y_pos2_train_indices
del Y_super_test_indices
del Y_super_train_indices

print('Saving prepared datal')

# save train/test/dev data

print('Saving test data', end ='')

np.savez('test.npz', X_test=X_test, X_test_embedding=X_test_embedding,\
         Y_pos1_test_oh=Y_pos1_test_oh, Y_pos2_test_oh=Y_pos2_test_oh, Y_super_test_oh=Y_super_test_oh)

print(' done!')

del X_test
del X_test_embedding
del Y_pos1_test_oh
del Y_pos2_test_oh
del Y_super_test_oh

print('Saving development data', end ='')

np.savez('dev.npz', X_dev=X_dev, X_dev_embedding=X_dev_embedding,\
         Y_pos1_dev_oh=Y_pos1_dev_oh, Y_pos2_dev_oh=Y_pos2_dev_oh, Y_super_dev_oh=Y_super_dev_oh)

print(' done!')

del X_dev
del X_dev_embedding
del Y_pos1_dev_oh
del Y_pos2_dev_oh
del Y_super_dev_oh

print('Saving training data', end ='')

np.savez('train.npz', X_train=X_train, X_train_embedding=X_train_embedding,\
         Y_pos1_train_oh=Y_pos1_train_oh, Y_pos2_train_oh=Y_pos2_train_oh, Y_super_train_oh=Y_super_train_oh)

print(' done!')

del X_train
del X_train_embedding
del Y_pos1_train_oh
del Y_pos2_train_oh
del Y_super_train_oh

