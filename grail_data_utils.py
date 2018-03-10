import matplotlib.pyplot as plt
import numpy as npy
import re


# normalizes word according to the conventions of fastText; only transforms the word to lower case

def normalize_word(orig_word):
    word = orig_word.lower()
    if (word is "["):
        word = "("
    if (word is "]"):
        word = ")"
    
    return word


def read_maxentdata(file):
    with open(file, 'r') as f:
        vocabulary = set()
        vnorm = set()
        partsofspeech1 = set()
        partsofspeech2 = set()
        superset = set()
        sentno = 0
        maxlen = 0
        words = []
        postags1 = []
        postags2 = []
        supertags = []
        allwords = []
        allpos1 = []
        allpos2 = []
        allsuper = []
        for line in f:
            line = line.strip().split()
            length = len(line)
            if (length > maxlen):
                maxlen = length
            for l in range(length):
                item = line[l].split('|')
                orig_word = item[0]
                word = normalize_word(orig_word)
                postag = item[1]
                supertag = item[2]
                poslist = postag.split('-')
                pos1 = poslist[0]
                pos2 = poslist[1]
                vocabulary.add(orig_word)
                vnorm.add(word)
                partsofspeech1.add(pos1)
                partsofspeech2.add(pos2)
                superset.add(supertag)
                words.append(orig_word)
                postags1.append(pos1)
                postags2.append(pos2)
                supertags.append(supertag)
            allwords.append(words)
            allpos1.append(postags1)
            allpos2.append(postags2)
            allsuper.append(supertags)
            words = []
            postags1 = []
            postags2 = []
            supertags = []
            
        X = npy.asarray(allwords)
        Y1 = npy.asarray(allpos1)
        Y2 = npy.asarray(allpos2)
        Z = npy.asarray(allsuper)
        return X, Y1, Y2, Z, vocabulary, vnorm, partsofspeech1, partsofspeech2, superset, maxlen


# create a bi-directional mapping (using two dictionaries) translating elements of a set to and from integers

def indexify (set):
    i = 1
    item_to_index = {}
    index_to_item = {}

    for item in set:
        item_to_index[item] = i
        index_to_item[i] = item
        i = i + 1

    return item_to_index, index_to_item


def is_numeral (string):
    return re.match(r'\A\-?\d[\d\.\,-/]*\Z', string) is not None

def num_to_vec (ns, number):
    vecsize = npy.size(number["0"])
    avg = npy.zeros(vecsize)

    cl = list(ns)
    i = 0
    for char in cl:
        avg += number[char]
        i += 1
    avg = avg/i
    return avg


def read_suffixes(file):
    i = 1
    suffixes = {}

    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            suffixes[line] = i
            i = i + 1

    return suffixes

french_suffixes = read_suffixes('suffixes.txt')

def suffix_vector(word, suffixes=french_suffixes):
    length = len(suffixes)+1
    vector = npy.zeros(length)
    for suf,num in suffixes.items():
        if word.endswith(suf):
            vector[num] = 1.0
        else:
            vector[num] = 0.0
    return vector

french_prepositions = set(['à', 'après', 'avant', 'avec', 'chez', 'contre', 'dans', 'de', 'dès', "d'", 'depuis', 'derrière', 'devant', 'en', 'entre', 'envers', 'hors', 'par', 'parmi', 'pendant', 'pour', 'sans', 'sauf', 'selon', 'sous', 'sur', 'vers'])

french_whq_words = set(['qui', 'que', "qu'", 'comment', 'combien', 'auquel', 'auxquels', 'auxquelles', 'duquel', 'desquels', 'desquelles','quel', 'quelle', 'quels', 'quelles', 'quand', 'où', 'pourquoi'])

french_definite_articles = set(['le', 'la', 'les', "l'", 'du', 'des', 'au', 'aux'])

french_possessive_articles = set(['mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leurs'])

french_indefinite_articles = set(['un', 'une'])

french_determiners =\
    set(['aucun', 'aucune', 'certaines', 'certains', 'plusieurs',\
         'quelque', "quelqu'", 'quelques',\
         'ladite', 'ledit', 'lesdits', 'lesdites', 'maint', 'maintes',\
         'tout', 'toute', 'tous', 'toutes',\
         'ma', 'mon', 'mes', 'ta', 'ton', 'tes', 'sa', 'son', 'ses',\
         'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',\
         'un', 'une',\
         'ce', 'ces', 'cet',  'cette',\
         "l'", 'la', 'le', 'les', 'chaque', "d'", 'de', 'des',\
         'du', 'des', 'au', 'aux'])

french_nominative_pronouns = set(['je', "j'", 'on', 'tu', "t'", 'il', 'elle', 'nous', 'vous', 'ils', 'elles',\
                                 '-t-il', '-t-ils', '-t-elle', '-t-elles', '-t-on', \
                                 '-je', '-tu', '-elle', '-il', '-on', '-nous', '-vous', '-ils', '-elles' ])

french_accusative_pronouns = set(['me', "m'", "-m'", '-moi', 'te', "t'", 'toi', '-toi', 'le', '-le', 'la', '-la', "l'", 'lui', '-lui',\
                                  'nous', '-nous', 'vous', '-vous', 'les', '-les', 'elles'])

french_reflexives = set(['se', "s'", 'me', "m'", 'te', "t'", 'toi', "nous", "vous"])

# all inflected forms of "être"

etre_forms = set(['suis', 'es', 'est', 'sommes', 'êtes', 'sont', \
                 'fus', 'fut', 'fûmes', 'fûtes', 'furent', \
                 'étais', 'était', 'étions', 'étiez', 'étaient', \
                 'serai', 'seras', 'sera', 'serons', 'serez', 'seront', \
                 'sois', 'soit', 'soyons', 'soyez', 'soient', \
                 'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent', \
                 'serais', 'serait', 'serions', 'seriez', 'seraient', \
                 'été'])

# all inflected forms of "avoir"

avoir_forms = set(['ai', 'as', 'a', 'avons', 'avez', 'ont', \
                  'eus', 'eut', 'eûmes', 'eûtes', 'eurent', \
                  'avais', 'avait', 'avions', 'aviez', 'avaient', \
                  'aurai', 'auras', 'aura', 'aurons', 'aurez', 'auront', \
                  'aie', 'aies', 'ait', 'ayons', 'ayez', 'aient', \
                  'eusse', 'eusses', 'eût', 'eussions', 'eussiez', 'eussent', \
                  'aurais', 'aurait', 'aurions', 'auriez', 'auraient', \
                   'eu'])

# past participles taking "être" as auxiliary verb

etre_past_participles = set(['allé', 'allée', 'allés', 'allées',\
                             'arrivé', 'arrivée', 'arrivés', 'arrivées',\
                             'descendu', 'descendue', 'descendus', 'descendues',\
                             'entré', 'entrée', 'entrés', 'entrées',\
                             'monté', 'montée', 'montés', 'montées',\
                             'mort', 'morte', 'morts', 'mortes',\
                             'né', 'née', 'nés', 'nées',\
                             'parti', 'partie', 'partis', 'parties',\
                             'passé', 'passée', 'passés', 'passées',\
                             'resté', 'restée', 'restés', 'restées',\
                             'retourné', 'retournée', 'retournés', 'retournées',\
                             'sorti', 'sortie', 'sortis', 'sorties',\
                             'tombé', 'tombée', 'tombés', 'tombées',\
                             'venu', 'venue', 'venus', 'venues',\
                             'devenu', 'devenue', 'devenus', 'devenues',\
                             'revenu', 'revenue', 'revenus', 'revenues'])
                             
                             
frequent_words = set([',', '-', ':', ';', '.', '?', '!', '...', '(', ')', '+',\
                      'face', 'affaire', 'assistance', 'audience', 'avis'\
                      'par', 'qui', 'dans', 'en', '%', 'sur', 'plus', '-',\
                      'pas', 'son', 'avec', 'beaucoup', 'confiance'\
                      'francs', 'ses', 'leur', 'comme', 'mais', 'pays', 'année', 'même', 'sa', 'ans', 'France', 'entre',\
                      'dont', 'fait', 'mois', 'groupe', 'depuis', 'marché', 'leurs', 'aussi', 'très', 'sans', 'tout', 'toute', 'toutes',  'prix', 'taux',\
                      'où', 'bien', 'après', 'moins', 'encore', 'contre', 'premier', 'autres', 'entreprises', 'faire', '?', ';', 'soit', 'peu',\
                      'an', 'temps', 'fin', 'début', 'pour', 'alors', 'années', 'ainsi', 'lui', 'tous', 'autre', 'peut', 'avant', 'selon', 'fois', 'déjà',\
                      'part', 'donc', 'quelques', 'sous', 'non', 'et', 'notre', 'devrait', 'cas', 'près', 'va', 'pourrait', "aujourd'hui",\
                      'effet', 'nombre', 'doit', 'étaient', 'toujours', 'vers', 'environ', 'faut', 'devant', 'surtout', 'autant', 'lors',\
                      'pouvoir', 'ailleurs', 'chaque', 'vie', 'raison', 'seulement', 'mis', 'aura', 'moment', 'nos', 'durée', 'aurait', 'partir',\
                      'conseil', 'ancien', 'dès', 'certains', 'chez', 'ici', 'moyenne', 'doute', 'nouvelles', 'ici', 'demande', 'lieu', 'pendant',\
                      'puis', 'jamais', 'cela', 'total', 'désormais', 'afin'])

def word_features(word, unknown=False):
    lcword = word.lower()
    list = []
    # word is in all-caps
    if word.isupper():
        list.append(1.0)
    else:
        list.append(0.0)
    # word starts with upper-case character    
    if word[0].isupper():
        list.append(1.0)
    else:
        list.append(0.0)
    # word is composed of [0-9]    
    if word.isnumeric():
        list.append(1.0)
    else:
        list.append(0.0)
    # word contains non-alphanumeric characters    
    if word.isalnum():
        list.append(0.0)
    else:
        list.append(1.0)
    # word starts with a hyphen    
    if word.startswith("-"):
         list.append(1.0)
    else:
        list.append(0.0)
    # word end with a hyphen    
    if word.endswith("-"):
         list.append(1.0)
    else:
        list.append(0.0)
    # word is a number (according to is_numeral)    
    if is_numeral(word):
         list.append(1.0)
    else:
        list.append(0.0)
    # hyphen in the middle of a word     
    if "-" in word[1:-1]:
        list.append(1.0)
    else:
        list.append(0.0)
    # word is an angular bracket (fasttext normalizes these away)    
    if (word is "[") or (word is "]"):
         list.append(1.0)
    else:
        list.append(0.0)
    # word is a quotation mark    
    if (word is "'") or (word is '"'):
         list.append(1.0)
    else:
        list.append(0.0)
    # word is form a "que"    
    if (lcword == "que") or (lcword == "qu'"):
         list.append(1.0)
    else:
        list.append(0.0)
    # word is form a "en"    
    if (lcword == "-t-en") or (lcword == "-en") or (lcword == "en"):
        list.append(1.0)
    else:
        list.append(0.0)
    # word is form a pronous "ce"    
    if (lcword == "-ce") or (lcword == "ce") or (lcword == "c'"):
        list.append(1.0)
    else:
        list.append(0.0)
    # word is form a determiner "ce"    
    if (lcword == "ces") or (lcword == "cette") or (lcword == "ce") or (lcword == "cet"):
        list.append(1.0)
    else:
        list.append(0.0)
    # word is form a demonstrative pronoun    
    if (lcword == "celle") or (lcword == "celui") or (lcword == "cela") or (lcword == "ceux"):
        list.append(1.0)
    else:
        list.append(0.0)
    # word is form a "y"    
    if (lcword == "-t-y") or (lcword == "-y") or (lcword == "y"):
        list.append(1.0)
    else:
        list.append(0.0)
    # word is form a "là"    
    if (lcword == "-là") or (lcword == "là"):
        list.append(1.0)
    else:
        list.append(0.0)
    # word is form a "à"    
    if (word == "A") or (lcword == "à") or (lcword == "au") or (lcword == "aux"):
        list.append(1.0)
    else:
        list.append(0.0)
    # word is form a "de"    
    if (lcword == "de") or (lcword == "du") or (lcword == "des") or (lcword == "d'"):
        list.append(1.0)
    else:
        list.append(0.0)
    # word is form a title    
    if (word == "M.") or (word == "Me") or (word == "Mme") or (word == "Mlle"):
        list.append(1.0)
    else:
        list.append(0.0)
    # word is form a "ne"    
    if (lcword == "ne") or (lcword == "n'"):
        list.append(1.0)
    else:
        list.append(0.0)
    # word is form a "jusque"    
    if (lcword == "jusque") or (lcword == "jusqu'"):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword in etre_forms):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword in etre_past_participles):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword in avoir_forms):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword in french_whq_words):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword in french_prepositions):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword in french_determiners):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword in french_definite_articles):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword in french_possessive_articles):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword in french_indefinite_articles):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword in french_nominative_pronouns):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword in french_accusative_pronouns):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword in french_reflexives):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword == "et") or (lcword == "ou"):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword == "si") or (lcword == "s'"):
        list.append(1.0)
    else:
        list.append(0.0)
    if (lcword == "pour") or (lcword == "contre"):
        list.append(1.0)
    else:
        list.append(0.0)
    for w in frequent_words:
        if (w == lcword):
            list.append(1.0)
        else:
            list.append(0.0)
                     
    # word is unknown by the embedding    
    if unknown:
         list.append(1.0)
    else:
        list.append(0.0)
    
    ar = npy.asarray(list)
    return ar


def read_vecs(file, vnorm, vocabulary):

    # fastText does not include native numbers; internally, these are translated into sequences of the words "zéro", "un", etc.
    # We add new entries averaging over the number symbols for numbers appearing in the French Treebank.
    with open("num_vec.txt", 'r') as f:
        number = {}
        for line in f:
            line = line.strip().split()
            numc = line[0]
            emb = npy.array(line[1:], dtype=npy.float64)
            number[numc] = emb
        
    with open(file, 'r') as f:
        words = set()
        vocabn = vnorm
        vocab = vocabulary
        word_to_vec_map = {}
        emsize = 0
        # special treatment for numerals
        numset = set()
        for w in vocabn:
            if is_numeral(w):
                numset.add(w)
                words.add(w)
                emb = num_to_vec(w, number)
                features = word_features(w)
                suf = suffix_vector(w)
                word_to_vec_map[w] = npy.concatenate((emb,suf,features))
        vocabn = vocabn.difference(numset)
        vocab = vocab.difference(numset)
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            if (curr_word in vocabn):
                vocabn.remove(curr_word)
                vocab.discard(curr_word)
                words.add(curr_word)                
                emb = npy.array(line[1:], dtype=npy.float64)
                emsize = npy.size(emb)
                features = word_features(curr_word)
                suf = suffix_vector(curr_word)
                word_to_vec_map[curr_word] = npy.concatenate((emb,suf,features))

        for w in vocab:
            words.add(w)
            wn = normalize_word(w)
            emb = npy.zeros(emsize)
            suf = suffix_vector(wn)
            try:
                vec = word_to_vec_map[wn]
            except:  
                print(w)
                features = word_features(w, unknown=True)
                word_to_vec_map[w] = npy.concatenate((emb,suf,features))
            else:
                emb = vec[0:emsize]
                features = word_features(w)
                word_to_vec_map[w] = npy.concatenate((emb,suf,features))

        for w in vocabn:
            words.add(w)
            try:
                vec = word_to_vec_map[w]
            except:  
                print(w)
                features = word_features(w, unknown=True)
                emb = npy.zeros(emsize)
                suf = suffix_vector(w)
                word_to_vec_map[w] = npy.concatenate((emb,suf,features))
        
                
        i = 2  # keep 1 for unknown
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def lists_to_indices(X, item_to_index, max_len, normalize=False):

    m = X.shape[0]                                   # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = npy.zeros((m,max_len))

    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split it into words. You should get a list of words.
        list = X[i]

        j = 0
        
        # Loop over the words of sentence_words
        for w in list:
            if normalize == True:
                w = normalize_word(w)
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            try:
                X_indices[i, j] = item_to_index[w]
            except:
                print("Unknown: ", w)
                X_indices[i, j] = 1  # unknown
            # Increment j to j + 1
            j = j + 1
            
    return X_indices

# GRADED FUNCTION: sentences_to_indices

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples
    
    ### START CODE HERE ###
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = npy.zeros((m,max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split it into words. You should get a list of words.
        sentence_words = X[i]
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # w = normalize_word(w)
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            try:
                X_indices[i, j] = word_to_index[w]
            except:
                print("Unknown: ", w)
                X_indices[i, j] = 1   # index for unknown words
            # Increment j to j + 1
            j = j + 1
            
    ### END CODE HERE ###
    
    return X_indices


def plot_confusion_matrix(y_actu, y_pred, title='Confusion matrix', cmap=plt.cm.gray_r):
    
    df_confusion = pd.crosstab(y_actu, y_pred.reshape(y_pred.shape[0],), rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = npy.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

def tag_sequence(sentence, model, wmap, imap, maxLen):
    list = sentence.strip().split()
    arr = npy.array([list])
    indices = lists_to_indices(arr, wmap, max_len = maxLen, normalize=True)
    pred = model.predict(indices)
    for j in range(len(list)):
        num = npy.argmax(pred[0][j])        
        print(list[j] + '|' + imap[num], end=' ')


def print_tagged(X, model, wmap, imap, maxLen):

    Xi = lists_to_indices(X, wmap, maxLen)
    pred = model.predict(Xi) 
    
    for i in range(len(X)-1):
        for j in range(len(X[i])):
            num = npy.argmax(pred[i][j])
            print(X[i][j] + '|' + imap[num], end = ' ')
        print()

def print_tagged_beta(X, model, beta, wmap, imap, maxLen):

    Xi = lists_to_indices(X, wmap, maxLen)
    pred = model.predict(Xi) 
    
    for i in range(len(X)-1):
        for j in range(len(X[i])):
            tags = predict_beta(pred[i][j],beta)
            print(X[i][j], end = '')
            print('|', end='')
            print(len(tags), end='')
            for key,value in tags.items():
                print('|', end='')
                print(imap[key], end='')
                print('|', end='')
                print(value, end='')
            print(' ', end='')    
        print()

# returns set with integer indices of all solutions with probability
# greater than or equal to beta time the probability assigned to the
# best solution
        
def predict_beta_set(vec,beta):
    tags = set()
    maxp = npy.max(vec)
    bm = maxp * beta
    for k in range(len(vec)):
        kprob = vec[k]
        if (kprob >= bm):
            tags.add(k)
    return tags

def predict_beta(vec,beta):
    tags = {}
    maxp = npy.max(vec)
    bm = maxp * beta
    for k in range(len(vec)):
        kprob = vec[k]
        if (kprob >= bm):
            tags[k] = kprob
    return tags

# This code allows you to see the mislabelled examples

def eval_beta(X_dev, Y_dev, model, wmap, imap, iimap, beta, maxLen):
    correct = 0
    wrong = 0
    totalpreds = 0

    Xi = lists_to_indices(X_dev, wmap, maxLen)
    Y_dev_indices = lists_to_indices(Y_dev, imap, max_len = maxLen)
    pred = model.predict(Xi)
    
    for i in range(len(X_dev)-1):
        for j in range(len(X_dev[i])):
            numset = predict_beta_set(pred[i][j], beta)
            totalpreds = totalpreds + len(numset)
            if not (Y_dev_indices[i][j] in numset):
                wrong = wrong + 1
                print('Expected tag: '+ X_dev[i][j] + '|' + Y_dev[i][j] + ' prediction: '+ X_dev[i][j],end='')
#                print(numset)
                for pi in numset:
                    print('|' + iimap[pi], end='')
                print()
            else:
                correct = correct + 1
    total = wrong + correct
    print("Total  : ", total)
    print("Correct: ", correct)
    print("Wrong  : ", wrong)

    cpct = (100*correct)/total
    wpct = (100*wrong)/total
    print("Correct %: ", cpct)
    print("Wrong   %: ", wpct)
    
    avpreds = totalpreds/total
    print("Average predictions : ", avpreds)


cardinal_map = {'un':1, 'une':1, 'deux':2, 'trois':3, 'quatre':4, 'cinq':5, 'six':6, 'sept':7, 'huit':8, 'neuf':9,\
                'dix':10, 'onze':11, 'douze':12, 'treize':13, 'quatorze':14, 'quinze':15, 'seize':16, 'dix-sept':17, 'dix-huit':18, 'dix-neuf':19,\
                'vingt':20, 'vingt-et-un':21, 'vingt-deux':22, 'vingt-trois':13, 'vingt-quatre':24, 'vingt-cinq':25, 'vingt-six':26, 'vingt-sept':27, 'vingt-huit':28, 'vingt-neuf':29,\
                'trente':30, 'trente-et-un':31, 'trente-deux':32, 'trente-trois':33, 'trente-quatre':34, 'trente-cinq':35, 'trente-six':36, 'trente-sept':37, 'trente-huit':38, 'trente-neuf':39,\
                'quarante':40, 'quarante-et-un':41, 'quarante-deux':32, 'quarante-trois':43, 'quarante-quatre':44, 'quarante-cinq':45, 'quarante-six':46, 'quarante-sept':47, 'quarante-huit':48, 'quarante-neuf':49,\
                'cinqante':50, 'cinquante-et-un':51, 'cinquante-deux':52, 'cinquante-trois':53, 'cinquante-quatre':54, 'cinquante-cinq':55, 'cinquante-six':56, 'cinquante-sept':57, 'cinquante-huit':58, 'cinquante-neuf':59,\
                'soixante':60, 'soixante-et-un':61, 'soixante-deux':62, 'soixante-trois':63, 'soixante-quatre':64, 'soixante-cinq':65, 'soixante-six':66, 'soixante-sept':67, 'soixante-huit':68, 'soixante-neuf':69,\
                'soixante-dix':70, 'soixante-et-onze':71, 'soixante-douze':72, 'soixante-treize':73, 'soixante-quatorze':74, 'soixante-quinze':75, 'soixante-seize':76, 'soixante-dix-sept':77, 'soixante-dix-huit':78, 'soixante-dix-neuf':79,\
                'quatre-vingts':80, 'quatre-vingt':80 , 'quatre-vingt-un':81, 'quatre-vingt-deux':82, 'quatre-vingt-trois':83, 'quatre-vingt-quatre':84, 'quatre-vingt-cinq':85, 'quatre-vingt-six':86, 'quatre-vingt-sept':87, 'quatre-vingt-huit':88, 'quatre-vingt-neuf':89,\
                'quatre-vingt-dix':90, 'quatre-vingt-onze':91, 'quatre-vingt-douze':92, 'quatre-vingt-treize':93, 'quatre-vingt-quatorze':94, 'quatre-vingt-quinze':95, 'quatre-vingt-seize':96, 'quatre-vingt-dix-sept':97, 'quatre-vingt-dix-huit':98, 'quatre-vingt-dix-neuf':99,\
                'cent':100, 'mille':1000, 'million':1000000, 'milliard':1000000 }
