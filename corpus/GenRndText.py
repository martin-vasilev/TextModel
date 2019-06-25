# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:47:07 2019

Generate text with random tokens

@author: Martin
"""

import os
import sys
import numpy as np
import random


os.chdir('C:\Github\TextModel\corpus')

with open('train.txt', 'r') as myfile:
    data= myfile.read()
    text= data.split('\n')

with open('vocab.txt', 'r') as myfile:
    data= myfile.read()
    vocab= data.split('\n')
    
with open('freq_occurance.txt', 'r') as myfile:
    data= myfile.read()
    freq= data.split('\n')
    
# get number of words per text piece in exisiting corpus:    
TextLen= []

for i in range(len(text)):
    words= text[i].split(' ')
    TextLen.append(len(words))

freq= list(map(int, freq))
    
Alltokens= np.sum(freq)

#rel_freq = [x / Alltokens for x in freq]

# create a 'word corpus' with frequency of occurance same as the real corpus:

Rnd_corpus= []

for i in range(len(vocab)):
    Rnd_corpus.extend(freq[i]*[vocab[i]])
    
assert(len(Rnd_corpus)== Alltokens)

############ Generate random text for images:
file = open("trainRND.txt", "w")

punct= [".", ",", "!", "?", ":", ";"]
numbers= ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

for i in range(len(TextLen)):
    tokens= random.sample(Rnd_corpus, TextLen[i])
    random.shuffle(tokens)
    string= ''
    
    for j in range(len(tokens)):
        curr_tkn= tokens[j]
        if curr_tkn in numbers:
            curr_tkn= '<num>'
        
        if curr_tkn in punct:
            string= string+ curr_tkn
        else:
            string= string+ ' '+ curr_tkn
            
    # write string to file 
    if string[0]== ' ':
        string= string[1:len(string)]
       
    file.write(string)
    if i<len(TextLen):
        file.write('\n')
    
file.close()


#########################################################################################################################
#                                                       Validate                                                        #
#########################################################################################################################

with open('validate.txt', 'r') as myfile:
    data= myfile.read()
    valid= data.split('\n')

TextLen= []

for i in range(len(valid)):
    words= valid[i].split(' ')
    TextLen.append(len(words))

############ Generate random text for images:
file = open("validRND.txt", "w")

for i in range(len(TextLen)):
    tokens= random.sample(Rnd_corpus, TextLen[i])
    random.shuffle(tokens)
    string= ''
    
    for j in range(len(tokens)):
        curr_tkn= tokens[j]
        if curr_tkn in numbers:
            curr_tkn= '<num>'
        
        if curr_tkn in punct:
            string= string+ curr_tkn
        else:
            string= string+ ' '+ curr_tkn
            
    # write string to file 
    if string[0]== ' ':
        string= string[1:len(string)]
       
    file.write(string)
    if i<len(TextLen):
        file.write('\n')
    
file.close()
    