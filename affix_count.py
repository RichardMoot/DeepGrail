#!/usr/bin/python3

import re

Cutoff = 5

def trim_dict(d, min_count=Cutoff):
    for k,v in list(d.items()):
        if v < min_count:
            del d[k]
    return d

file=open("test.txt","r+")
wordcount={}
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

for word in file.read().strip().split():
    word = word.lower()
    word = re.sub(r'[0-8]', '9', word)
    suf1 = word[-1:]
    suf2 = word[-2:]
    suf3 = word[-3:]
    suf4 = word[-4:]
    suf5 = word[-5:]
    suf6 = word[-6:]
    suf7 = word[-7:]
    pref1 = word [:1]
    pref2 = word [:2]
    pref3 = word [:3]
    pref4 = word [:4]
    
    if word not in wordcount:
        wordcount[word] = 1
    else:
        wordcount[word] += 1

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

file.close();

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


#for k,v in list(prefixcount4.items()):
#    if v < cutoff:
#        del prefixcount4[k]

print(prefix4)
print(suffix7)
print(len(prefix2)+len(prefix3)+len(prefix4)+len(suffix1)+len(suffix2)+len(suffix3)+len(suffix4)+len(suffix5)+len(suffix6)+len(suffix7))
