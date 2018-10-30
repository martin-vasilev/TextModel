# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:54:40 2018

@author: mvasilev
"""
import os
import numpy as np
from Corpus import Corpus
#from itertools import chain
os.chdir('C:\\Users\\mvasilev\\TextModel\\corpus')

file = open("good_checked.txt", "w")
file_why= open("why.txt", "w")

with open("good.txt", 'r') as myfile:
    data= myfile.read()
    text= data.split('\n')
text = filter(None, text)
text= Corpus.unique_list(text)

tokens= Corpus.SUBTLEX(20000) # get N SUBTLEX tokens

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
    
    if len(out)>0:
        print(i)
        for j in range(len(out)):
            file_why.write(out[j])
            file_why.write(" ")
        file_why.write("\n")
    else:
        if len(string)>360:
            file.write(string)
            file.write("\n")
        else:
            print(i)
            file_why.write("LENGTH")
            file_why.write("\n")
    if len(string)> 360*2:
        print("yes")
        
file.close()
file_why.close()