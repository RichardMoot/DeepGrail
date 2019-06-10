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

np.random.seed(1)

best_file = 'best_elmo_superpos.h5'

# load corpus data
print('Loading training/dev data')

trainf = np.load('train.npz')

X_train = trainf['X_train']
X_train_embedding = trainf['X_train_embedding']
Y_pos1_train_oh = trainf['Y_pos1_train_oh']
Y_pos2_train_oh = trainf['Y_pos2_train_oh']
Y_super_train_oh = trainf['Y_super_train_oh']

del trainf

devf = np.load('dev.npz')

X_dev = devf['X_dev']
X_dev_embedding = devf['X_dev_embedding']
Y_pos1_dev_oh = devf['Y_pos1_dev_oh']
Y_pos2_dev_oh = devf['Y_pos2_dev_oh']
Y_super_dev_oh = devf['Y_super_dev_oh']

# compute constants from array sizes

ns, maxLen, embLen = np.shape(X_dev_embedding)
tmp1, tmp2, numPos1Classes = np.shape(Y_pos1_dev_oh)
tmp1, tmp2, numPos2Classes = np.shape(Y_pos2_dev_oh)
tmp1, tmp2, numSuperClasses = np.shape(Y_super_dev_oh)

del devf

# define model

sentence_embeddings = Input(shape = (maxLen,embLen,), dtype = 'float32')
mask = Masking(mask_value=0.0)(sentence_embeddings)
X = Dropout(0.5)(mask)

# first bi-directional LSTM layer 

X = Bidirectional(LSTM(128, recurrent_dropout=0.2, kernel_constraint=max_norm(5.), return_sequences=True))(X)
X = BatchNormalization()(X)
X = Dropout(0.2)(X)

# Pos1 output

Pos1 = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
Pos1 = TimeDistributed(Dropout(0.2))(Pos1)
Pos1 = TimeDistributed(Dense(numPos1Classes, name='pos1', activation='softmax',kernel_constraint=max_norm(5.)))(Pos1)

# Pos2 output

Pos2 = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
Pos2 = TimeDistributed(Dropout(0.2))(Pos2)
Pos2 = TimeDistributed(Dense(numPos2Classes, name='pos2', activation='softmax',kernel_constraint=max_norm(5.)))(Pos2)

# second bi-directional LSTM layer

X = Bidirectional(LSTM(128, recurrent_dropout=0.2, kernel_constraint=max_norm(5.), return_sequences=True))(X)
X = BatchNormalization()(X)
X = Dropout(0.2)(X)

# supertag output

X = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
X = TimeDistributed(Dropout(0.2))(X)
X = TimeDistributed(Dense(numSuperClasses, name='super', activation='softmax',kernel_constraint=max_norm(5.)))(X)

model = Model(sentence_embeddings, [Pos1,Pos2,X])

model.summary()

model.compile(loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'], loss_weights=[0.15,0.35,0.5], optimizer='rmsprop', metrics=['accuracy'])

checkpoint = ModelCheckpoint(best_file, monitor='val_time_distributed_9_acc', verbose=1, save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,\
                                              verbose=1,patience=5, min_lr=0.0001)


history = model.fit([X_train_embedding],\
          [Y_pos1_train_oh,Y_pos2_train_oh,Y_super_train_oh],\
          epochs=30, shuffle=True, batch_size=32,\
          callbacks = [checkpoint,reduce_lr],\
          validation_data=(X_dev_embedding,\
                           [Y_pos1_dev_oh,Y_pos2_dev_oh,Y_super_dev_oh]))
