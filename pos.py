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

def load_all():
    load_obj(word_to_index, 'word_to_index')
    load_obj(index_to_word, 'index_to_word')
    load_obj(word_to_prefix, 'word_to_prefix')
    load_obj(prefix_to_word, 'prefix_to_word')
    load_obj(word_to_suffix, 'word_to_suffix')
    load_obj(suffix_to_word, 'suffix_to_word')
    load_obj(pos1_to_index, 'pos1_to_index')
    load_obj(index_to_pos1, 'index_to_pos1')
    load_obj(pos2_to_index, 'pos2_to_index')
    load_obj(index_to_pos2, 'index_to_pos2')
    load_obj(word_to_vec_map, 'word_to_vec_map')
    load_obj(p1_to_integer, 'p1_to_integer')
    load_obj(integer_to_p1, 'integer_to_p1')
    load_obj(p2_to_integer, 'p2_to_integer')
    load_obj(integer_to_21, 'integer_to_p2')
    load_obj(p3_to_integer, 'p3_to_integer')
    load_obj(integer_to_31, 'integer_to_p3')
    load_obj(p4_to_integer, 'p4_to_integer')
    load_obj(integer_to_p4, 'integer_to_p4')
    load_obj(s1_to_integer, 's1_to_integer')
    load_obj(integer_to_s1, 'integer_to_s1')
    load_obj(s2_to_integer, 's2_to_integer')
    load_obj(integer_to_s2, 'integer_to_s2')
    load_obj(s3_to_integer, 's3_to_integer')
    load_obj(integer_to_s3, 'integer_to_s3')
    load_obj(s4_to_integer, 's4_to_integer')
    load_obj(integer_to_s4, 'integer_to_s4')
    load_obj(s5_to_integer, 's5_to_integer')
    load_obj(integer_to_s5, 'integer_to_s5')
    load_obj(s6_to_integer, 's6_to_integer')
    load_obj(integer_to_s6, 'integer_to_s6')
    load_obj(s7_to_integer, 's7_to_integer')
    load_obj(integer_to_s7, 'integer_to_s7')


load_all()
model = load_model('best_pos/pos.h5')

def read_text_file(filename):
    with open(filename, 'r') as f:
        lines = 0
        maxlen = 0
        text = {}
        for line in f:
            line = line.strip().split()
            length = len(line)
            if (length > maxlen):
                maxlen = length
            text[lines] = line
            lines = lines + 1
    return text, maxlen

def missing_vocab(text, vocab):
    missing = set()
    for (k,v) in text.items():
        for i in rangle(len(v)):
            word = v[i]
            if word not in vocab:
                missing.add(word)
    return missing

text, maxline = read_text_file('input.txt')

vocab = word_to_index.keys()

missing = missing_vocab(text, vocab)

# update word_to_index and index_to_word

lenv = max(vocab, key=int)
lenv = lenv + 1
for w in missing:
    word_to_index[w] = lenv
    index_to_word[lenv] = w
    lenv = lenv + 1

# update affixes

word_to_prefix_plus, word_to_suffix_plus = compute_affixes(missing)
word_to_prefix.update(word_to_prefix_plus)
word_to_suffix.update(word_to_suffix_plus)

# update embeddings

wv = KeyedVectors.load_word2vec_format('../wang2vec/frwiki_cwindow50_10.bin', binary=True)
veclength = 50

word_to_vec_map = {}
unknowns = set()
invoc = 0

for w in missing:
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

X = sentences_to_indices(text, word_to_index, maxlen)
