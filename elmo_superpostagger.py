#!/usr/local/bin/python3

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

from grail_data_utils import *

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

# load precomputed ELMo array
files = np.load('Xarr.npz')
Xarr = files['Xarr']

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

print('Compiling model')
# define the model

sentence_embeddings = Input(shape = (maxLen,1024,), dtype = 'float32')
mask = Masking(mask_value=0.0)(sentence_embeddings)

# get probability distribution over parts_of_speech from pos_model
X = Dropout(0.5)(mask)
        
# Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
# returning a batch of sequences.
X = Bidirectional(LSTM(128, recurrent_dropout=0.2, kernel_constraint=max_norm(5.), return_sequences=True))(X) 
X = BatchNormalization()(X)
X = Dropout(0.4)(X)
    
Pos1 = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
Pos1 = TimeDistributed(Dropout(0.5))(Pos1)
Pos1 = TimeDistributed(Dense(numPos1Classes, name='pos1', activation='softmax',kernel_constraint=max_norm(5.)))(Pos1)

Pos2 = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
Pos2 = TimeDistributed(Dropout(0.5))(Pos2)
Pos2 = TimeDistributed(Dense(numPos2Classes, name='pos2', activation='softmax',kernel_constraint=max_norm(5.)))(Pos2)

X = Bidirectional(LSTM(128, recurrent_dropout=0.2, kernel_constraint=max_norm(5.), return_sequences=True))(X) 
X = BatchNormalization()(X)
X = Dropout(0.4)(X)

# Add a (time distributed) Dense layer followed by a softmax activation
X = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
X = TimeDistributed(Dropout(0.4))(X)
X = TimeDistributed(Dense(numSuperClasses, name='super', activation='softmax',kernel_constraint=max_norm(5.)))(X)

model = Model(sentence_embeddings, [Pos1,Pos2,X])
model.summary()

# compile and train

model.compile(optimizer='rmsprop', loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'], loss_weights=[0.15,0.35,0.5],  metrics=['accuracy'])



best_file = "best_elmo_superpos.h5"
checkpoint = ModelCheckpoint(best_file, monitor='val_time_distributed_9_acc', verbose=1, save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,\
                                              verbose=1,patience=5, min_lr=0.0001)


history =model.fit([X_train_embedding],\
          [Y_pos1_train_oh,Y_pos2_train_oh,Y_super_train_oh],\
          epochs=30, shuffle=True, batch_size=32,\
          callbacks = [checkpoint,reduce_lr],\
          validation_data=(X_dev_embedding,\
                           [Y_pos1_dev_oh,Y_pos2_dev_oh,Y_super_dev_oh]))
