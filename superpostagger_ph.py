import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle

from keras.models import Model, load_model
from keras.layers import Bidirectional, Dense, Input, Dropout, LSTM, Activation, TimeDistributed, BatchNormalization, concatenate, Concatenate, Masking
from keras.layers.embeddings import Embedding
from keras.constraints import max_norm
from keras import regularizers
from keras import optimizers
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

from grail_data_utils import *
from lstm_peephole import *

np.random.seed(1)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

super_to_index = load_obj('super_to_index')
index_to_super = load_obj('index_to_super')
pos1_to_index = load_obj('pos1_to_index')
index_to_pos1 = load_obj('index_to_pos1')
pos2_to_index = load_obj('pos2_to_index')
index_to_pos2 = load_obj('index_to_pos2')
p1_to_integer = load_obj('p1_to_integer')
integer_to_p1 = load_obj('integer_to_p1')
p2_to_integer = load_obj('p2_to_integer')
integer_to_p2 = load_obj('integer_to_p2')
p3_to_integer = load_obj('p3_to_integer')
integer_to_p3 = load_obj('integer_to_p3')
p4_to_integer = load_obj('p4_to_integer')
integer_to_p4 = load_obj('integer_to_p4')
s1_to_integer = load_obj('s1_to_integer')
integer_to_s1 = load_obj('integer_to_s1')
s2_to_integer = load_obj('s2_to_integer')
integer_to_s2 = load_obj('integer_to_s2')
s3_to_integer = load_obj('s3_to_integer')
integer_to_s3 = load_obj('integer_to_s3')
s4_to_integer = load_obj('s4_to_integer')
integer_to_s4 = load_obj('integer_to_s4')
s5_to_integer = load_obj('s5_to_integer')
integer_to_s5 = load_obj('integer_to_s5')
s6_to_integer = load_obj('s6_to_integer')
integer_to_s6 = load_obj('integer_to_s6')
s7_to_integer = load_obj('s7_to_integer')
integer_to_s7 = load_obj('integer_to_s7')

maxLen = 266
numSuperClasses = len(index_to_super) + 1
numPos1Classes = len(index_to_pos1) + 1
numPos2Classes = len(index_to_pos2) + 1

X, Y1, Y2, Z, vocabulary, vnorm,\
    partsofspeech1, partsofspeech2, superset, maxLen = read_maxentdata('m2.txt')

X_train, X_testdev, Y_pos1_train, Y_pos1_testdev, Y_pos2_train, Y_pos2_testdev, Y_super_train, Y_super_testdev = train_test_split(X, Y1, Y2, Z, test_size=0.4)
X_test, X_dev, Y_pos1_test, Y_pos1_dev, Y_pos2_test, Y_pos2_dev, Y_super_test, Y_super_dev = train_test_split(X_testdev, Y_pos1_testdev, Y_pos2_testdev, Y_super_testdev, test_size=0.5)
print("Train: ", X_train.shape)
print("Test:  ", X_test.shape)
print("Dev:   ", X_dev.shape)

del X_testdev
del Y_pos1_testdev
del Y_pos2_testdev
del Y_super_testdev

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

print()
print("Reading word embeddings")

wv = KeyedVectors.load_word2vec_format('../wang2vec/frwiki_cwindow50_10.bin', binary=True)
veclength = 50

word_to_vec_map = {}
unknowns = set()
invoc = 0

for w in vocabulary:
    wn = normalize_word(w)
    wr = remove_prefix(wn, "-t-")
    wr = remove_prefix(wr, "-")
    try:
        vec = wv[wr]
        invoc = invoc + 1
    except:
        unknowns.add(w)
        vec = np.zeros(veclength)
    word_to_vec_map[w] = vec

print('Unknowns: ', len(unknowns))
print('In vocabulary: ', invoc)

def word_to_prefvec(word, alen, afset, af_to_int):
    if len(word) >= alen:
        pref = word[:alen]
        if pref in afset:
            int = af_to_int[pref]
        else:
            int = af_to_int['*UNK*']
    else:
        int = af_to_int['*OOR*']
    return to_categorical(int, len(afset)+1)


def word_to_sufvec(word, alen, afset, af_to_int):
    if len(word) >= alen:
        pref = word[-alen:]
        if pref in afset:
            int = af_to_int[pref]
        else:
            int = af_to_int['*UNK*']
    else:
        int = af_to_int['*OOR*']
    return to_categorical(int, len(afset)+1)

prefix1 = p1_to_integer.keys()
prefix2 = p2_to_integer.keys()
prefix3 = p3_to_integer.keys()
prefix4 = p4_to_integer.keys()

suffix1 = s1_to_integer.keys()
suffix2 = s2_to_integer.keys()
suffix3 = s3_to_integer.keys()
suffix4 = s4_to_integer.keys()
suffix5 = s5_to_integer.keys()
suffix6 = s6_to_integer.keys()
suffix7 = s7_to_integer.keys()

def word_to_prefix_vector(word):
    p1 = word_to_prefvec(word, 1, prefix1, p1_to_integer)
    p2 = word_to_prefvec(word, 2, prefix2, p2_to_integer)
    p3 = word_to_prefvec(word, 3, prefix3, p3_to_integer)
    p4 = word_to_prefvec(word, 4, prefix4, p4_to_integer)
    return np.concatenate((p1,p2,p3,p4))

def word_to_suffix_vector(word):
    s1 = word_to_sufvec(word, 1, suffix1, s1_to_integer)
    s2 = word_to_sufvec(word, 2, suffix2, s2_to_integer)
    s3 = word_to_sufvec(word, 3, suffix3, s3_to_integer)
    s4 = word_to_sufvec(word, 4, suffix4, s4_to_integer)
    s5 = word_to_sufvec(word, 5, suffix5, s5_to_integer)
    s6 = word_to_sufvec(word, 6, suffix6, s6_to_integer)
    s7 = word_to_sufvec(word, 7, suffix7, s7_to_integer)
    return np.concatenate((s1,s2,s3,s4,s5,s6,s7))

def word_to_affix_vector(word):
    p1 = word_to_prefvec(word, 1, prefix1, p1_to_integer)
    p2 = word_to_prefvec(word, 2, prefix2, p2_to_integer)
    p3 = word_to_prefvec(word, 3, prefix3, p3_to_integer)
    p4 = word_to_prefvec(word, 4, prefix3, p4_to_integer)
    s1 = word_to_sufvec(word, 1, suffix1, s1_to_integer)
    s2 = word_to_sufvec(word, 2, suffix2, s2_to_integer)
    s3 = word_to_sufvec(word, 3, suffix3, s3_to_integer)
    s4 = word_to_sufvec(word, 4, suffix4, s4_to_integer)
    s5 = word_to_sufvec(word, 5, suffix5, s5_to_integer)
    s6 = word_to_sufvec(word, 6, suffix6, s6_to_integer)
    s7 = word_to_sufvec(word, 7, suffix7, s7_to_integer)
    return np.concatenate((p1,p2,p3,p4,s1,s2,s3,s4,s5,s6,s7))

def compute_affixes(vocab):
    
    word_to_suffix = {}
    word_to_prefix = {}

    for word in vocab:
        w = word.lower()
        w = re.sub(r'[0-8]', '9', w)
        pvec = word_to_prefix_vector(w)
        svec = word_to_suffix_vector(w)
        word_to_prefix[word] = pvec
        word_to_suffix[word] = svec
        
    return word_to_prefix, word_to_suffix

word_to_index, index_to_word = indexify(vocabulary)
word_to_prefix, word_to_suffix = compute_affixes(vocabulary)


X_train_indices = lists_to_indices(X_train, word_to_index, maxLen)
X_dev_indices = lists_to_indices(X_dev, word_to_index, max_len = maxLen)

Y_super_train_indices = lists_to_indices(Y_super_train, super_to_index, maxLen)
Y_super_train_oh = to_categorical(Y_super_train_indices, num_classes=numSuperClasses)
# supertag results
Y_super_train_indices = lists_to_indices(Y_super_train, super_to_index, maxLen)
Y_super_train_oh = to_categorical(Y_super_train_indices, num_classes=numSuperClasses)

Y_super_dev_indices = lists_to_indices(Y_super_dev, super_to_index, max_len = maxLen)
Y_super_dev_oh = to_categorical(Y_super_dev_indices, num_classes = numSuperClasses)

del Y_super_dev
del Y_super_dev_indices

# pos1 results
Y_pos1_train_indices = lists_to_indices(Y_pos1_train, pos1_to_index, maxLen)
Y_pos1_train_oh = to_categorical(Y_pos1_train_indices, num_classes=numPos1Classes)

Y_pos1_dev_indices = lists_to_indices(Y_pos1_dev, pos1_to_index, max_len = maxLen)
Y_pos1_dev_oh = to_categorical(Y_pos1_dev_indices, num_classes = numPos1Classes)

del Y_pos1_dev
del Y_pos1_dev_indices

# pos2 results
Y_pos2_train_indices = lists_to_indices(Y_pos2_train, pos2_to_index, maxLen)
Y_pos2_train_oh = to_categorical(Y_pos2_train_indices, num_classes=numPos2Classes)


del Y_pos2_train
del Y_pos2_train_indices

Y_pos2_dev_indices = lists_to_indices(Y_pos2_dev, pos2_to_index, max_len = maxLen)
Y_pos2_dev_oh = to_categorical(Y_pos2_dev_indices, num_classes = numPos2Classes)


del Y_pos2_dev
del Y_pos2_dev_indices



def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained fastText vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 2                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["est"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len,emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(vocab_len,emb_dim,trainable=False,mask_zero=True)

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


sentence_indices = Input(shape = (maxLen,), dtype = 'int32')

embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
prefix_emb = pretrained_embedding_layer(word_to_prefix, word_to_index)
suffix_emb = pretrained_embedding_layer(word_to_suffix, word_to_index)

pref = prefix_emb(sentence_indices)
suff = suffix_emb(sentence_indices)
P = Dense(32)(pref)

S = Dense(32)(suff)
embeddings = embedding_layer(sentence_indices)

merged = concatenate([embeddings,P,S])

X = Dropout(0.5)(merged)
X = Bidirectional(PeepholeLSTM(128, recurrent_dropout=0.2, kernel_constraint=max_norm(5.), return_sequences=True))(X) 
X = BatchNormalization()(X)
X = Dropout(0.2)(X)

Pos1 = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
Pos1 = TimeDistributed(Dropout(0.2))(Pos1)
Pos1 = TimeDistributed(Dense(numPos1Classes, name='pos1', activation='softmax',kernel_constraint=max_norm(5.)))(Pos1)

Pos2 = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
Pos2 = TimeDistributed(Dropout(0.2))(Pos2)
Pos2 = TimeDistributed(Dense(numPos2Classes, name='pos2', activation='softmax',kernel_constraint=max_norm(5.)))(Pos2)

X = Bidirectional(PeepholeLSTM(128, recurrent_dropout=0.2, kernel_constraint=max_norm(5.), return_sequences=True))(X) 
X = BatchNormalization()(X)
X = Dropout(0.2)(X)
    # Add a 1d convolution to make predictions dependent on context
    # X = Conv1D(64, 5, padding='same', kernel_constraint=max_norm(5.))(X)
    # Add a (time distributed) Dense layer followed by a softmax activation
X = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
X = TimeDistributed(Dropout(0.2))(X)
X = TimeDistributed(Dense(numSuperClasses, name='super', activation='softmax',kernel_constraint=max_norm(5.)))(X)

model = Model(sentence_indices, [Pos1,Pos2,X])
model.summary()

model.compile(optimizer='rmsprop', loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'], loss_weights=[0.15,0.35,0.5], metrics=['accuracy'])

#model.fit([X_train_emb,X_train_pref,X_train_suff], [Y_pos1_train_oh,Y_pos2_train_oh,Y_train_dev_oh], epochs=10, batch_size=32)

# train on dev set (smaller)
#model.fit([X_dev_indices], [Y_pos1_dev_oh,Y_pos2_dev_oh,Y_super_dev_oh], epochs=10, batch_size=32, validation_data=(X_train_indices[:2000],[Y_pos1_train_oh[:2000],Y_pos2_train_oh[:2000],Y_super_train_oh[:2000]]))

best_file = "best_super.h5"
checkpoint = ModelCheckpoint(best_file, monitor='val_time_distributed_9_acc', verbose=1, save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,\
                                              verbose=1,patience=5, min_lr=0.0001)

history =model.fit([X_train_indices],\
          [Y_pos1_train_oh,Y_pos2_train_oh,Y_super_train_oh],\
          epochs=30, shuffle=True, batch_size=32,\
          callbacks = [checkpoint,reduce_lr],\
          validation_data=(X_dev_indices,\
                           [Y_pos1_dev_oh,Y_pos2_dev_oh,Y_super_dev_oh]))
