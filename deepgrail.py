import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle

from keras.models import Model, load_model
from keras.layers import Bidirectional, Dense, Input, Dropout, LSTM, Activation, TimeDistributed, BatchNormalization, concatenate, Concatenate
from keras.layers.embeddings import Embedding
from keras.constraints import max_norm
from keras import regularizers
from keras import optimizers
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

from grail_data_utils import *


np.random.seed(1)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_all():
    save_obj(word_to_index, 'word_to_index')
    save_obj(index_to_word, 'index_to_word')
    save_obj(super_to_index, 'super_to_index')
    save_obj(index_to_super, 'index_to_super')
    save_obj(word_to_vec_map, 'word_to_vec_map')
    save_obj(p1_to_integer, 'p1_to_integer')
    save_obj(integer_to_p1, 'integer_to_p1')
    save_obj(p2_to_integer, 'p2_to_integer')
    save_obj(integer_to_21, 'integer_to_p2')
    save_obj(p3_to_integer, 'p3_to_integer')
    save_obj(integer_to_31, 'integer_to_p3')
    save_obj(p4_to_integer, 'p4_to_integer')
    save_obj(integer_to_p4, 'integer_to_p4')
    save_obj(s1_to_integer, 's1_to_integer')
    save_obj(integer_to_s1, 'integer_to_s1')
    save_obj(s2_to_integer, 's2_to_integer')
    save_obj(integer_to_s2, 'integer_to_s2')
    save_obj(s3_to_integer, 's3_to_integer')
    save_obj(integer_to_s3, 'integer_to_s3')
    save_obj(s4_to_integer, 's4_to_integer')
    save_obj(integer_to_s4, 'integer_to_s4')
    save_obj(s5_to_integer, 's5_to_integer')
    save_obj(integer_to_s5, 'integer_to_s5')
    save_obj(s6_to_integer, 's6_to_integer')
    save_obj(integer_to_s6, 'integer_to_s6')
    save_obj(s7_to_integer, 's7_to_integer')
    save_obj(integer_to_s7, 'integer_to_s7')

# very small initial part of corpus (only file aa1)
# X, Y1, Y2, Z, vocabulary, vnorm, partsofspeech1, partsofspeech2, superset, maxLen = read_maxentdata('aa1.txt')

# small initial part of corpus (files aa1, aa2, ab2 and ae1)
# number of sentences, train: 1195, test: 398, dev: 399  
# X, Y1, Y2, Z, vocabulary, vnorm, partsofspeech1, partsofspeech2, superset, maxLen = read_maxentdata('aa1_ae1.txt')

# entire corpus
# number of sentences, train: 9449, test: 3150, dev: 3150
X, Y1, Y2, Z, vocabulary, vnorm,\
    partsofspeech1, partsofspeech2, superset, maxLen = read_maxentdata('m2.txt')

# X_train = X_train[:2048]
numClasses = len(partsofspeech2)+1
numSuperClasses = len(superset)+1

print()
print("Longest sentence   : ", maxLen)
print("Number of POS tags : ", numClasses)
print("Number of supertags: ", numSuperClasses)

X_train, X_testdev, Y_super_train, Y_super_testdev = train_test_split(X, Z, test_size=0.4)
X_test, X_dev, Y_super_test, Y_super_dev = train_test_split(X_testdev, Y_super_testdev, test_size=0.5)
print("Train: ", X_train.shape)
print("Test:  ", X_test.shape)
print("Dev:   ", X_dev.shape)
# create mapping for the two POS tagset and for the supertags

super_to_index, index_to_super = indexify(superset)

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

word_to_index, index_to_word = indexify(vocabulary)


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
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

print("Computing affixes")

Cutoff = 2

def trim_dict(d, min_count=Cutoff):
    for k,v in list(d.items()):
        if v < min_count:
            del d[k]
    d['*UNK*'] = 1
    d['*OOR*'] = 1
    return d

suffixcount1={}
suffixcount2={}
suffixcount3={}
suffixcount4={}
suffixcount5={}
suffixcount6={}
suffixcount7={}
prefixcount1={}
prefixcount2={}
prefixcount3={}
prefixcount4={}

for word in vocabulary:
    # convert to lower case and replace all digits by '9'
    word = word.lower()
    word = re.sub(r'[0-8]', '9', word)
    # take suffixes of length 7 or smaller (to distinguish between 'eraient' and 'aient')    
    suf1 = word[-1:]
    suf2 = word[-2:]
    suf3 = word[-3:]
    suf4 = word[-4:]
    suf5 = word[-5:]
    suf6 = word[-6:]
    suf7 = word[-7:]
    # take prefixes of length 4 or smaller    
    pref1 = word [:1]
    pref2 = word [:2]
    pref3 = word [:3]
    pref4 = word [:4]
    
    # update all counters for the computed affixes
    if len(suf1) > 0:
        if suf1 not in suffixcount1:
            suffixcount1[suf1] = 1
        else:
            suffixcount1[suf1] += 1

    if len(suf2) > 1: 
        if suf2 not in suffixcount2:
            suffixcount2[suf2] = 1
        else:
            suffixcount2[suf2] += 1

    if len(suf3) > 2: 
        if suf3 not in suffixcount3:
            suffixcount3[suf3] = 1
        else:
            suffixcount3[suf3] += 1

    if len(suf4) > 3: 
        if suf4 not in suffixcount4:
            suffixcount4[suf4] = 1
        else:
            suffixcount4[suf4] += 1

    if len(suf5) > 4: 
        if suf5 not in suffixcount5:
            suffixcount5[suf5] = 1
        else:
            suffixcount5[suf5] += 1
    if len(suf6) > 5: 
        if suf6 not in suffixcount6:
            suffixcount6[suf6] = 1
        else:
            suffixcount6[suf6] += 1
    if len(suf7) > 6: 
        if suf7 not in suffixcount7:
            suffixcount7[suf7] = 1
        else:
            suffixcount7[suf7] += 1
    if len(pref1) > 0:
        if pref1 not in prefixcount1:
            prefixcount1[pref1] = 1
        else:
            prefixcount1[pref1] += 1

    if len(pref2) > 1:
        if pref2 not in prefixcount2:
            prefixcount2[pref2] = 1
        else:
            prefixcount2[pref2] += 1

    if len(pref3) > 2:
        if pref3 not in prefixcount3:
            prefixcount3[pref3] = 1
        else:
            prefixcount3[pref3] += 1
    if len(pref4) > 3:
        if pref4 not in prefixcount4:
            prefixcount4[pref4] = 1
        else:
            prefixcount4[pref4] += 1


suffixcount1 = trim_dict(suffixcount1)
suffixcount2 = trim_dict(suffixcount2)
suffixcount3 = trim_dict(suffixcount3)
suffixcount4 = trim_dict(suffixcount4)
suffixcount5 = trim_dict(suffixcount5)
suffixcount6 = trim_dict(suffixcount6)
suffixcount7 = trim_dict(suffixcount7)

prefixcount1 = trim_dict(prefixcount1)
prefixcount2 = trim_dict(prefixcount2)
prefixcount3 = trim_dict(prefixcount3)
prefixcount4 = trim_dict(prefixcount4)

suffix1 = set(suffixcount1.keys())
suffix2 = set(suffixcount2.keys())
suffix3 = set(suffixcount3.keys())
suffix4 = set(suffixcount4.keys())
suffix5 = set(suffixcount5.keys())
suffix6 = set(suffixcount6.keys())
suffix7 = set(suffixcount7.keys())

prefix1 = set(prefixcount1.keys())
prefix2 = set(prefixcount2.keys())
prefix3 = set(prefixcount3.keys())
prefix4 = set(prefixcount4.keys())


p1_to_integer, integer_to_p1 = indexify(prefix1)
p2_to_integer, integer_to_p2 = indexify(prefix2)
p3_to_integer, integer_to_p3 = indexify(prefix3)
p4_to_integer, integer_to_p4 = indexify(prefix4)

s1_to_integer, integer_to_s1 = indexify(suffix1)
s2_to_integer, integer_to_s2 = indexify(suffix2)
s3_to_integer, integer_to_s3 = indexify(suffix3)
s4_to_integer, integer_to_s4 = indexify(suffix4)
s5_to_integer, integer_to_s5 = indexify(suffix5)
s6_to_integer, integer_to_s6 = indexify(suffix6)
s7_to_integer, integer_to_s7 = indexify(suffix7)

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

word_to_prefix, word_to_suffix = compute_affixes(vocabulary)

print("Compiling model")


# Super_model
# this is a direct supertag model not using the part-of-speech tags

def Super_affix_model(input_shape, word_to_vec_map, word_to_prefix, word_to_suffix, word_to_index):
    """
    Function creating the direct supertagger model's graph
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its fastText vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary

    Returns:
    model -- a model instance in Keras
    """
    
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape = input_shape, dtype = 'int32')
    
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    prefix_emb = pretrained_embedding_layer(word_to_prefix, word_to_index)
    suffix_emb = pretrained_embedding_layer(word_to_suffix, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)   
    
    pref = prefix_emb(sentence_indices)
    suff = suffix_emb(sentence_indices)
    P = Dense(32)(pref)
    S = Dense(32)(suff)
    merged = concatenate([embeddings,P,S])
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # returning a batch of sequences.
    X = Dropout(0.5)(merged)
    X = Bidirectional(LSTM(128, recurrent_dropout=0.2, kernel_constraint=max_norm(5.), return_sequences=True))(X) 
    X = BatchNormalization()(X)
    X = Dropout(0.2)(X)
    X = Bidirectional(LSTM(128, recurrent_dropout=0.2, kernel_constraint=max_norm(5.), return_sequences=True))(X) 
    X = BatchNormalization()(X)
    X = Dropout(0.2)(X)
    # Add a 1d convolution to make predictions dependent on context
    # X = Conv1D(64, 5, padding='same', kernel_constraint=max_norm(5.))(X)
    # Add a (time distributed) Dense layer followed by a softmax activation
    X = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
    X = TimeDistributed(Dropout(0.2))(X)
    X = TimeDistributed(Dense(numSuperClasses, activation='softmax',kernel_constraint=max_norm(5.)))(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices,outputs=X)
        
    return model

print("Preparing training data")


X_train_indices = lists_to_indices(X_train, word_to_index, maxLen)
Y_super_train_indices = lists_to_indices(Y_super_train, super_to_index, maxLen)
Y_super_train_oh = to_categorical(Y_super_train_indices, num_classes=numSuperClasses)

print("Preparing development data")

X_dev_indices = lists_to_indices(X_dev, word_to_index, max_len = maxLen)
Y_super_dev_indices = lists_to_indices(Y_super_dev, super_to_index, max_len = maxLen)
Y_super_dev_oh = to_categorical(Y_super_dev_indices, num_classes = numSuperClasses)

supermodel = Super_affix_model((maxLen,), word_to_vec_map, word_to_prefix, word_to_suffix, word_to_index)
supermodel.summary()


# the main problem with the previous settings appeared to be slow convergence.
# set the learning rate to 0.005 (instead of the default 0.001)

adam_opt = optimizers.Adam(lr=0.005)
supermodel.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])

best_file = "best_super.h5"
checkpoint = keras.callbacks.ModelCheckpoint(best_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,\
                              patience=5, min_lr=0.0001)

history = supermodel.fit(X_train_indices, Y_super_train_oh,\
                         epochs = 50, batch_size = 32, shuffle=True,\
                         callbacks = [checkpoint,reduce_lr],
                         validation_data=(X_dev_indices,Y_super_dev_oh))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model train vs validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#supermodel.save('super.h5')
