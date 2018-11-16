# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:00:46 2018

@author: marti
"""

from Corpus import Corpus
import os
os.chdir('D:\\R\\TextModel\\corpus')

file = open("wiki_clean.txt", "w", encoding='utf-8')

with open("wiki_corpus2.txt", 'r', encoding='utf-8') as myfile:
    data= myfile.read()
text= data.split('\n')


text = list(filter(None, text)) # remove empty lines
text= Corpus.unique_list(text) # remove duplicates

for i in range(len(text)):
    if len(text[i])>= 6*60:
        file.write(text[i])
        file.write('\n')
file.close()