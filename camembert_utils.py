import numpy as np

camembert_lex_file = '/Users/moot/checkout/DeepGrail/camembert.v0/dict.txt'

def read_camembert_lex(file=camembert_lex_file):
    camembert_lex = {}
    idx = 4
    with open(file, 'r') as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            camembert_lex[idx] = word
            idx = idx + 1

    return camembert_lex

def print_tokens(tokens, lex):
    for i in range(len(tokens)):
        id = tokens[i]
        w = lex[id]
        print(w, end=' ')
    print('')

def merge_words(words, tokens, array, emblen):
    j = 0
    arlen = len(words)
    lex = read_camembert_lex()
    print_tokens(tokens, lex)
    out = np.zeros((1,arlen,emblen))
    for i in range(arlen):
        w = words[i]
        t = tokens[j]
        if t==26 and w != '-':
            j = j + 1
            t = tokens[j]
        out[0][i] = array[0][j]
        j = j + 1
    if j != len(tokens):
        print('Warning: untreated tokens', end=' ')
        print(j, end=' ')
        print(len(tokens))
    return out
