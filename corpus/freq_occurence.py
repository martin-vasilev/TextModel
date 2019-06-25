# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:06:49 2019

@author: Martin
"""

import os
import sys

os.chdir('C:\Github\TextModel')
from core.Corpus import Corpus
os.chdir('C:\Github\TextModel\corpus')
import numpy as np
from collections import Counter

# Python code to count the number of occurrences 
def countX(lst, x): 
    count = 0
    for ele in lst: 
        if (ele == x): 
            count = count + 1
    return count 


with open('train.txt', 'r') as myfile:
    data= myfile.read()
    text= data.split('\n')

with open('vocab.txt', 'r') as myfile:
    data= myfile.read()
    vocab= data.split('\n')
    
#merged_text= ' '.join(text)
#tokens_text= merged_text.split(' ')

Alltokens= []

for i in range(len(text)):
    wrds= Corpus.strip2(text[i])
    Alltokens.extend(wrds)
    if i%100==0:
        print(i)

corpus_freq= []


for i in range(len(vocab)):
    corpus_freq.append(countX(Alltokens, vocab[i]))
    if i%100==0:
        print(i)
        
file = open("freq_occurance.txt", "w")

for i in range(len(corpus_freq)):
    file.write(str(corpus_freq[i]))
    file.write('\n')
file.close()