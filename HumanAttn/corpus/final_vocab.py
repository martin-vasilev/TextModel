# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:39:25 2018

@author: mvasilev
"""

import os
os.chdir('D:\\Github\\TextModel')
import numpy as np
#import numpy as np
from core.Corpus import Corpus
#from itertools import chain

used_words= []

max_len= 0
#file = open("corpus_final_checked.txt", "w")

with open("corpus\\all_corpus.txt", 'r') as myfile:
    data= myfile.read()
    text= data.split('\n')
text = list(filter(None, text))
text= Corpus.unique_list(text)
tokens= Corpus.SUBTLEX(N= 20000, dir= "corpus\\SUBTLEX-US.txt")

for i in range(len(text)):
    string= text[i]
    
    newstr= string.replace("n't", " not")

    if "-" in newstr: # for compound words
        newstr= newstr.split("-") 
        newstr= ' '.join(newstr)
    if "/" in newstr:
        newstr= newstr.split("/")
        newstr= ' '.join(newstr)
    
    wrds= Corpus.strip(newstr)
    out = np.setdiff1d(wrds, tokens)
    out= out.tolist()
    
    lsr = [x for x in wrds if x not in out]
    used_words.extend(lsr)
    
    if len(wrds) > max_len:
        max_len= len(wrds)
    
    if len(out)>0:
        print("unrecognized token %s" % out)
    if i % 100 == 0:
        print(i)
    
used_words= Corpus.unique_list(used_words) 

tokensN= l3 = [x for x in tokens if x in used_words]   

file = open("vocab.txt", "w")

for i in range(len(tokensN)):
    file.write(tokensN[i])
    file.write('\n')
    
file.close()

print(max_len)