# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:10:24 2018

@author: mvasilev
"""

import os
import numpy as np
from Corpus import Corpus

os.chdir('C:\\Users\\mvasilev\\TextModel\\corpus')
tokens= Corpus.SUBTLEX(20000) # get N SUBTLEX tokens

file = open("good.txt", "w")
result= open("result.txt", "w")

with open("COCA_news.txt", 'r') as myfile:
    data= myfile.read()
    text= data.split('\n')

for i in range(len(text)):
    string= text[i]
    wrds= Corpus.strip(string)
    out = np.setdiff1d(wrds, tokens)
    out= out.tolist()
    
    if len(out)>0:
        for j in range(len(out)):
            result.write(out[j])
            result.write(" ")
        result.write("\n")
    else:
        result.write("0")
        result.write("\n")
        
        file.write(string)
        file.write("\n")
        
file.close()
result.close()