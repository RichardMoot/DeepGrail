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
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard
from keras.utils import to_categorical
from keras import backend as K
from sklearn.model_selection import train_test_split

np.random.seed(1)

epochs = 10
current_file = 'current_elmo_superpos.h5'
best_file = 'best_elmo_superpos.h5'

try:
    opts, args = getopt.getopt(sys.argv[1:],"me",["epochs=","model="])
    print(opts)
except getopt.GetoptError as err:
    print(str(err))
    print("elmo_continue.py -b <beta_value> -i <inputfile> -o <outputfile> -m <modelfile>")

for opt, arg in opts:
    if opt == "-h":
        print("elmo_continue.py -e <epochs> -m <modelfile>")
    elif opt in ("-m", "--model"):
        current_file = arg
    elif opt in ("-e", "--epoch", "--epochs"):
        epochs = int(arg)

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

model = load_model(current_file)

checkpoint = ModelCheckpoint(best_file, monitor='val_time_distributed_9_acc', verbose=1, save_best_only=True, mode='max')

save_current = ModelCheckpoint(current_file, monitor='val_time_distributed_9_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,\
                                              verbose=1,patience=5, min_lr=0.0001)

log = CSVLogger('elmo_training_log.csv', append=True)


history = model.fit([X_train_embedding],\
          [Y_pos1_train_oh,Y_pos2_train_oh,Y_super_train_oh],\
          epochs=epochs, shuffle=True, batch_size=32,\
          callbacks = [checkpoint,reduce_lr,save_current,log],\
          validation_data=(X_dev_embedding,\
                           [Y_pos1_dev_oh,Y_pos2_dev_oh,Y_super_dev_oh]))
