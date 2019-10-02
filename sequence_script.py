import numpy as np
import tensorflow as tf
import os, random

from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Bidirectional, Lambda, Masking, Dense, Input, Dropout, LSTM, Activation, TimeDistributed, BatchNormalization, concatenate, Concatenate
from keras.layers.embeddings import Embedding
from keras.constraints import max_norm, min_max_norm, unit_norm
from keras import regularizers
from keras.initializers import random_uniform
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard
from keras.utils import to_categorical
from keras import backend as K
from all_elmo_sequence import DataGenerator

# set random seed to seed_value for reproducability

seed_value = 1
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

# parameters for generator class

params = {'n_pos1_classes':30,
          'n_pos2_classes':32,
          'n_super_classes':891,
          'shuffle':True,
          'batch_size':32}

# filenames for best and last model files

best_file = 'best_gen_elmo_superpos.h5'
current_file = 'current_gen_elmo_superpos.h5'

# number of sentences in the treebank; presupposes the existence of file "sent%06d.npz"
# for i from 0 to treebank_sentences-1 in the TLGbank directory

treebank_sentences = 15748

all = ["sent%06d" %i for i in range(treebank_sentences)]

# standard 60/20/20 split for train/dev/test

train, testdev = train_test_split(all, test_size=0.4)
test, dev = train_test_split(testdev, test_size=0.5)

print("Train: "+str(len(train)))
print("Dev  : "+str(len(dev)))
print("Test : "+str(len(test)))

training_generator = DataGenerator(train, **params)
validation_generator = DataGenerator(dev, **params)

embLen = 1024
numPos1Classes = 30
numPos2Classes = 32
numSuperClasses = 891

l1_value = 0.0001
l2_value = 0.0001

# input layers are the three ELMo output layers

sentence_embeddings0 = Input(shape = (None,embLen,), dtype = 'float32')
sentence_embeddings1 = Input(shape = (None,embLen,), dtype = 'float32')
sentence_embeddings2 = Input(shape = (None,embLen,), dtype = 'float32')

# take weighted average of three inputs

stacked = Lambda(lambda x: K.stack([x[0],x[1],x[2]], axis=-1))([sentence_embeddings0,sentence_embeddings1,sentence_embeddings2])
# use unit_norm to force sum of 1 and intialize weights close to 0.33
#weighted = Dense(1, kernel_constraint=unit_norm(), kernel_initializer=random_uniform(0.30,0.36), use_bias=False)(stacked)
# use weighted sum with l1l2 regularization
weighted = Dense(1, kernel_regularizer=regularizers.l1_l2(l1_value,l2_value), use_bias=False)(stacked)
weighted = Lambda(lambda x: K.squeeze(x, axis=-1))(weighted)

#concat = concatenate([sentence_embeddings0, sentence_embeddings1, sentence_embeddings2])

mask = Masking(mask_value=0.0)(weighted)
dropout = Dropout(0.5)(mask)

# first bi-directional LSTM layer 

X = Bidirectional(LSTM(128, recurrent_dropout=0.2, kernel_constraint=max_norm(4.), return_sequences=True))(mask)
X = BatchNormalization()(X)
X = Dropout(0.25)(X)

# Pos1 output

Pos1 = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
Pos1 = TimeDistributed(Dropout(0.25))(Pos1)
pos1_output = TimeDistributed(Dense(numPos1Classes, name='pos1_output', activation='softmax',kernel_constraint=max_norm(4.)))(Pos1)

# Pos2 output

Pos2 = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
Pos2 = TimeDistributed(Dropout(0.25))(Pos2)
pos2_output = TimeDistributed(Dense(numPos2Classes, name='pos2_output', activation='softmax',kernel_constraint=max_norm(4.)))(Pos2)

# second bi-directional LSTM layer

X = Bidirectional(LSTM(128, recurrent_dropout=0.25, kernel_constraint=max_norm(4.), return_sequences=True))(X)
X = BatchNormalization()(X)
X = Dropout(0.25)(X)
# concatenate ELMo vectors before output; doesn't improve performance
# X = concatenate([X,dropout])

# supertag output

X = TimeDistributed(Dense(32,kernel_regularizer=regularizers.l1_l2(l1_value,l2_value)))(X)
X = TimeDistributed(Dropout(0.25))(X)
super_output = TimeDistributed(Dense(numSuperClasses, name='super_output', activation='softmax',kernel_regularizer=regularizers.l1_l2(l1_value,l2_value)))(X)

model = Model([sentence_embeddings0, sentence_embeddings1, sentence_embeddings2], [pos1_output,pos2_output,super_output])

model.summary()

model.compile(loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'], optimizer='rmsprop', metrics=['accuracy'])

checkpoint = ModelCheckpoint(best_file, monitor='val_time_distributed_9_acc', verbose=1, save_best_only=True, mode='max')

save_current = ModelCheckpoint(current_file, monitor='val_time_distributed_9_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,\
                                              verbose=1,patience=5, min_lr=0.0001)

log = CSVLogger('elmo_training_log.csv')


history = model.fit_generator(training_generator,\
                              epochs=100, shuffle=True, workers=2, use_multiprocessing=True,\
                              callbacks = [checkpoint,reduce_lr,log,save_current],
                              validation_data=validation_generator)


