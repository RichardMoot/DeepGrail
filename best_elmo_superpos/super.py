#!/Users/moot/anaconda3/bin/python

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
from elmoformanylangs import Embedder

from grail_data_utils import *

inputfile = 'input.txt'
outputfile = 'super.txt'
beta = 1.0
modelfile = 'best_elmo_superpos.h5'

try:
    opts, args = getopt.getopt(sys.argv[1:],"hbiom",["beta=","input=","output=","model="])
    print(opts)
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



current_dir = os.getcwd()
os.chdir('/Users/moot/checkout/DeepGrail/best_elmo_superpos')   


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

        
# load ELMo embedder

print('Loading French ELMo embeddings')

e = Embedder('/Users/moot/Software/FrenchELMo/')

# load corpus data

def read_text_file(filename):
    with open(filename, 'r') as f:
        lines = 0
        text = []
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
                text.append(outwords)
                if outpos != []:
                    pos[lines] = outpos
                lines = lines + 1
    return text, pos, lines

print('Reading input file')

text, pos, numLines = read_text_file(inputfile)

print('Computing French ELMo embeddings for input text')

text_emb = e.sents2elmo(text)

ll = len(text_emb)
Xarr= np.zeros((ll,maxLen,1024))
for i in range(ll):
    sl = len(text[i])
    for j in range(sl):
        Xarr[i][j]= text_emb[i][j]

sentence_embeddings = Input(shape = (maxLen,1024,), dtype = 'float32')

# # get probability distribution over parts_of_speech from pos_model
# X = Dropout(0.5)(sentence_embeddings)
        
# # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
# # returning a batch of sequences.
# X = Bidirectional(LSTM(128, recurrent_dropout=0.2, kernel_constraint=max_norm(5.), return_sequences=True))(X) 
# X = BatchNormalization()(X)
# X = Dropout(0.2)(X)
    
# Pos1 = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
# Pos1 = TimeDistributed(Dropout(0.2))(Pos1)
# Pos1 = TimeDistributed(Dense(numPos1Classes, name='pos1', activation='softmax',kernel_constraint=max_norm(5.)))(Pos1)

# Pos2 = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
# Pos2 = TimeDistributed(Dropout(0.2))(Pos2)
# Pos2 = TimeDistributed(Dense(numPos2Classes, name='pos2', activation='softmax',kernel_constraint=max_norm(5.)))(Pos2)

# X = Bidirectional(LSTM(128, recurrent_dropout=0.2, kernel_constraint=max_norm(5.), return_sequences=True))(X) 
# X = BatchNormalization()(X)
# X = Dropout(0.2)(X)

# # Add a (time distributed) Dense layer followed by a softmax activation
# X = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
# X = TimeDistributed(Dropout(0.2))(X)
# X = TimeDistributed(Dense(numSuperClasses, name='super', activation='softmax',kernel_constraint=max_norm(5.)))(X)

# model = Model(sentence_embeddings, [Pos1,Pos2,X])
# model.summary()
        
model = load_model(modelfile)
predict_pos1, predict_pos2, predict_super = model.predict(Xarr)

os.chdir(current_dir)

f = open(outputfile, 'w')

index_to_super[0] = '*UNK*'

for i in range(len(text)):
    string = ""
    for j in range(len(text[i])):
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
        wrd = text[i][j]
        string = string + " " + wrd +'|'+posstr+tagstr
    string = string.strip()
    print(string)
    string = string + "\n"
    f.write(string)

f.close()
exit()
