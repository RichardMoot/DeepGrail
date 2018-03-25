#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, getopt
import os, os.path
import pickle
import operator

from keras.models import Model, load_model
from keras.layers import Bidirectional, Dense, Input, Dropout, LSTM, Activation, TimeDistributed, BatchNormalization, concatenate, Concatenate
from keras.layers.embeddings import Embedding
from keras.constraints import max_norm
from keras import regularizers
from keras.utils import to_categorical
from keras import backend as K
from gensim.models import KeyedVectors

from grail_data_utils import *

inputfile = 'input.txt'
outputfile = 'super.txt'
beta = 0.01
modelfile = 'superposmodel.h5'

try:
    opts, args = getopt.getopt(sys.argv[1:],"hbiom",["beta=","input=","output=","model="])
except getopt.GetoptError as err:
    print(str(err))
    print("super.py -b <beta_value> -i <inputfile> -o <outputfile> -m <modelfile>")

for opt, arg in opts:
    if opt == "-h":
        print("super.py -b <beta_value> -i <inputfile> -o <outputfile> -m <modelfile>")
    elif opt in ("-m", "--model"):
        modelfile = arg
    elif opt in ("-i", "--input"):
        inputfile = arg
    elif opt in ("-o", "--output"):
        outputfile = arg
    elif opt in ("-b", "--beta"):
        beta = float(arg)


          
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

current_dir = os.getcwd()
os.chdir('/Users/moot/checkout/DeepGrail/best_superpos')   

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


def read_text_file(filename):
    with open(filename, 'r') as f:
        lines = 0
        text = {}
        pos = {}
        for line in f:
            outwords = []
            outpos = []
            line = line.strip().split()
            length = len(line)
            if (length > maxLen):
                print("Skipped long sentence (", end='')
                print(length, end='')
                print("):")
                print(line)
            else:
                for i in range(length):
                    item = line[i]
                    iitems = item.split('|')
                    word = iitems[0]
                    outwords.append(word)
                    if len(iitems) > 1:
                        ipos = iitems[1]
                        outpos.append(ipos)
                text[lines] = outwords
                if outpos != []:
                    pos[lines] = outpos
                lines = lines + 1
    return text, pos, lines

def text_vocab(text):
    vocab = set()
    pos = {}
    for (k,v) in text.items():
        for i in range(len(v)):
            word = v[i]
            if word not in vocab:
                vocab.add(word)
    return vocab

text, pos, numLines = read_text_file(inputfile)

vocab = text_vocab(text)

word_to_index, index_to_word = indexify(vocab)

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

word_to_prefix, word_to_suffix = compute_affixes(vocab)


wv = KeyedVectors.load_word2vec_format('../../wang2vec/frwiki_cwindow50_10.bin', binary=True)
veclength = 50

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

word_to_vec_map = {}
unknowns = set()
invoc = 0

for w in vocab:
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

X_indices = np.zeros((numLines,266))

for i in range(numLines):
    line = text[i]
    for j in range(len(line)):
        word = line[j]
        X_indices[i,j] = word_to_index[word]

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained fastText vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 2           # adding 1 for 'unknown'and 1 to fit Keras embedding
    # get embedding size from first element
    key = list(word_to_vec_map.keys())[0]
    emb_dim = word_to_vec_map[key].shape[0]    # get dimensionality of word vectors
    
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
X = Bidirectional(LSTM(128, recurrent_dropout=0.2, kernel_constraint=max_norm(5.), return_sequences=True))(X) 
X = BatchNormalization()(X)
X = Dropout(0.2)(X)

Pos1 = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
Pos1 = TimeDistributed(Dropout(0.2))(Pos1)
Pos1 = TimeDistributed(Dense(numPos1Classes, name='pos1', activation='softmax',kernel_constraint=max_norm(5.)))(Pos1)

Pos2 = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
Pos2 = TimeDistributed(Dropout(0.2))(Pos2)
Pos2 = TimeDistributed(Dense(numPos2Classes, name='pos2', activation='softmax',kernel_constraint=max_norm(5.)))(Pos2)

X = Bidirectional(LSTM(128, recurrent_dropout=0.2, kernel_constraint=max_norm(5.), return_sequences=True))(X) 
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


trained_model = load_model(modelfile)

weights = trained_model.get_weights()

weights2 = model.get_weights()

for i in range(3,len(weights)):
    weights2[i] = weights[i]

model.set_weights(weights2)

predict_pos1, predict_pos2, predict_super = model.predict(X_indices)

os.chdir(current_dir)

f = open(outputfile, 'w')

for i in range(len(X_indices)):
    string = ""
    for j in range(len(X_indices[i]-1)):
        if X_indices[i][j] != 0:
            if pos != {}:
                sentpos = pos[i]
                jpos = sentpos[j]
                posstr = str(jpos) + "|"
            else:
                pos1num = np.argmax(predict_pos1[i][j])
                pos2num = np.argmax(predict_pos2[i][j])
                posstr = index_to_pos1[pos1num] + "-" + index_to_pos2[pos2num] + "|"
            if beta < 1:
                tags = predict_beta(predict_super[i][j],beta)
                tagstr = str(len(tags))
                while tags != {}:
                    cmax = max(tags.items(), key=operator.itemgetter(1))[0]
                    pstr = str(tags[cmax])
                    del tags[cmax]
                    tstr = str(index_to_super[cmax])
                    tagstr = tagstr + "|" + tstr + "|" + pstr
            else:
                num = np.argmax(predict_super[i][j])
                tagstr = str(index_to_super[num])
            wi = int(X_indices[i][j])
            string = string + " " + str(index_to_word[wi])+'|'+posstr+tagstr
    string = string.strip()
    print(string)
    string = string + "\n"
    f.write(string)

f.close()
exit()
