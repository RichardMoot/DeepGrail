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
            
        X = np.asarray(allwords)
        Y1 = np.asarray(allpos1)
        Y2 = np.asarray(allpos2)
        Z = np.asarray(allsuper)
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

def num_to_vec (ns):
    vecsize = np.size(number["0"])
    avg = np.zeros(vecsize)

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

def suffix_vector(word, suffixes=french_suffixes):
    length = len(suffixes)+1
    vector = np.zeros(length)
    for suf,num in suffixes.items():
        if word.endswith(suf):
            vector[num] = 1.0
        else:
            vector[num] = 0.0
    return vector

french_prepositions = set(['à', 'après', 'avant', 'avec', 'chez', 'contre', 'dans', 'de', 'dès', "d'", 'depuis', 'derrière', 'devant', 'en', 'entre', 'envers', 'hors', 'par', 'parmi', 'pendant', 'pour', 'sans', 'sauf', 'selon', 'sous', 'sur', 'vers'])

french_whq_words = set(['qui', 'que', "qu'", 'comment', 'combien', 'auquel', 'auxquels', 'auxquelles', 'duquel', 'desquels', 'desquelles','quel', 'quelle', 'quels', 'quelles', 'quand', 'où', 'pourquoi'])

french_definite_articles = set(['le', 'la', 'les', "l'", 'du', 'des', 'au', 'aux'])

french_indefinite_articles = set(['un', 'une'])

french_nominative_pronouns = set(['je', "j'", 'on', 'tu', "t'", 'il', 'elle', 'nous', 'vous', 'ils', 'elles',\
                                 '-t-il', '-t-ils', '-t-elle', '-t-elles', '-t-on', \
                                 '-je', '-tu', '-elle', '-il', '-on', '-nous', '-vous', '-ils', '-elles' ])

french_accusative_pronouns = set(['me', "m'", 'te', "t'", 'toi', 'le', 'la', "l'", 'lui', 'nous', 'vous', 'les', 'elles'])

french_reflexives = set(['se', "s'", 'me', "m'", 'te', "t'", 'toi', "nous", "vous"])

etre_forms = set(['suis', 'es', 'est', 'sommes', 'êtes', 'sont', \
                 'fus', 'fut', 'fûmes', 'fûtes', 'furent', \
                 'étais', 'était', 'étions', 'étiez', 'étaient', \
                 'serai', 'seras', 'sera', 'serons', 'serez', 'seront', \
                 'sois', 'soit', 'soyons', 'soyez', 'soient', \
                 'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent', \
                 'serais', 'serait', 'serions', 'seriez', 'seraient', \
                 'été'])

avoir_forms = set(['ai', 'as', 'a', 'avons', 'avez', 'ont', \
                  'eus', 'eut', 'eûmes', 'eûtes', 'eurent', \
                  'avais', 'avait', 'avions', 'aviez', 'avaient', \
                  'aurai', 'auras', 'aura', 'aurons', 'aurez', 'auront', \
                  'aie', 'aies', 'ait', 'ayons', 'ayez', 'aient', \
                  'eusse', 'eusses', 'eût', 'eussions', 'eussiez', 'eussent', \
                  'aurais', 'aurait', 'aurions', 'auriez', 'auraient', \
                   'eu'])

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
    if (lcword in etre_forms):
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
    if (lcword in french_definite_articles):
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
    if (word == "A") or (lcword == "à") or (lcword == "au"):
         list.append(1.0)
    else:
        list.append(0.0)        
    if (lcword == "de") or (lcword == "d'") or (lcword == "du"):
         list.append(1.0)
    else:
        list.append(0.0)        
    # word is unknown by the embedding    
    if unknown:
         list.append(1.0)
    else:
        list.append(0.0)
    
    ar = np.asarray(list)
    return ar


def read_vecs(file):
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
                emb = num_to_vec(w)
                features = word_features(w)
                suf = suffix_vector(w)
                word_to_vec_map[w] = np.concatenate((emb,suf,features))
        vocabn = vocabn.difference(numset)
        vocab = vocab.difference(numset)
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            if (curr_word in vocabn):
                vocabn.remove(curr_word)
                vocab.discard(curr_word)
                words.add(curr_word)                
                emb = np.array(line[1:], dtype=np.float64)
                emsize = np.size(emb)
                features = word_features(curr_word)
                suf = suffix_vector(curr_word)
                word_to_vec_map[curr_word] = np.concatenate((emb,suf,features))

        for w in vocab:
            words.add(w)
            wn = normalize_word(w)
            emb = np.zeros(emsize)
            suf = suffix_vector(wn)
            try:
                vec = word_to_vec_map[wn]
            except:  
                print(w)
                features = word_features(w, unknown=True)
                word_to_vec_map[w] = np.concatenate((emb,suf,features))
            else:
                emb = vec[0:emsize]
                features = word_features(w)
                word_to_vec_map[w] = np.concatenate((emb,suf,features))

        for w in vocabn:
            words.add(w)
            try:
                vec = word_to_vec_map[w]
            except:  
                print(w)
                features = word_features(w, unknown=True)
                emb = np.zeros(emsize)
                suf = suffix_vector(w)
                word_to_vec_map[w] = np.concatenate((emb,suf,features))
        
                
        i = 2  # keep 1 for unknown
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

