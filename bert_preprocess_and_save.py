#!/Users/moot/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys, getopt
import os, os.path
import pickle
import operator
import sys

from sklearn.model_selection import train_test_split

from grail_data_utils import *

import torch
from transformers import FlaubertModel, FlaubertTokenizer

modelname = 'flaubert-large-cased' 

# Load pretrained model and tokenizer
flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=False)
# do_lowercase=False if using cased models, True if using uncased ones

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

numSuperClasses = len(index_to_super) + 3
numPos1Classes = len(index_to_pos1) + 3
numPos2Classes = len(index_to_pos2) + 3

bos_super = numSuperClasses - 1
bos_pos1 = numPos1Classes - 1
bos_pos2 = numPos2Classes - 1


super_to_index['<BOS>'] = bos_super
pos1_to_index['<BOS>'] = bos_pos1
pos2_to_index['<BOS>'] = bos_pos2

index_to_super[bos_super] = '<BOS>'
index_to_pos1[bos_pos1] = '<BOS>'
index_to_pos2[bos_pos2] = '<BOS>'

eos_super = numSuperClasses - 2
eos_pos1 = numPos1Classes - 2
eos_pos2 = numPos2Classes - 2

super_to_index['<EOS>'] = eos_super
pos1_to_index['<EOS>'] = eos_pos1
pos2_to_index['<EOS>'] = eos_pos2

index_to_super[eos_super] = '<EOS>'
index_to_pos1[eos_pos1] = '<EOS>'
index_to_pos2[eos_pos2] = '<EOS>'

# load corpus data
print('Loading corpus data')

# X, Y1, Y2, Z, vocabulary, vnorm, partsofspeech1, partsofspeech2, superset, maxLen = read_maxentdata('aa1.txt')
X, Y1, Y2, Z, vocabulary, vnorm, partsofspeech1, partsofspeech2, superset, maxLen = read_maxentdata('m2.txt')

print(np.shape(X))

# computing BERT array

cdir = "./BERT_TLGbank/"

for cursent in range(len(X)-1):

    fname = "sent%06d.npz" % cursent
    file = os.path.normpath(cdir + fname)

    bwords = X[cursent]
    words = ['<BOS>'] + bwords + ['<EOS>']

    pos1 = ['<BOS>'] + Y1[cursent] + ['<EOS>']
    pos1 = list_to_indices(pos1, pos1_to_index)

    pos2 = ['<BOS>'] + Y2[cursent] + ['<EOS>']
    pos2 = list_to_indices(pos2, pos2_to_index)

    super = ['<BOS>'] + Z[cursent] + ['<EOS>']
    super = list_to_indices(super, super_to_index)

    token_ids = torch.tensor([flaubert_tokenizer.encode(bwords)])
    last_layer = flaubert(token_ids)[0]
    bert_embedding = last_layer.detach()
    bert_numpy = bert_embedding.numpy()
#    bert_numpy = bert_numpy[:,1:-1,:]

    l1,l2,l3 = bert_numpy.shape
    if l2 != len(words):
        sys.exit("Error: "+ bwords)
        
    np.savez(file, words=words, pos1=pos1, pos2=pos2, super=super, bert=bert_numpy)


