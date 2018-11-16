# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:10:24 2018

@author: mvasilev
"""

minChar= 6*60

import os
os.chdir('D:\\R\\TextModel\\corpus')
import numpy as np
from Corpus import Corpus
from itertools import chain


tokens= Corpus.SUBTLEX(20000, 'SUBTLEX-US.txt') # get N SUBTLEX tokens

file = open("good_wiki.txt", "w", encoding='utf-8')
result= open("result_wiki.txt", "w", encoding='utf-8')

#file = open("good.txt", "a")
#result= open("result.txt", "a")

#with open("COCA_news.txt", 'r') as myfile:
with open("wiki_clean.txt", 'r', encoding='utf-8') as myfile:
    data= myfile.read()
    text= data.split('\n')

for i in range(len(text)):
    string= text[i]
    wrds= Corpus.strip(string)
    out = np.setdiff1d(wrds, tokens)
    out= out.tolist()
    string= string.replace(" n't", "n't")
    
    if len(out)>0:
        for j in range(len(out)):
            result.write(out[j])
            result.write(" ")
        result.write("\n")
        
        # check if string can still be saved:
        noList= []
        wrds= Corpus.strip(string, lower= False)
        for m in range(len(out)):
            rs= Corpus.find_substr(out[m], string)
            if len(rs)==0:
                rs= Corpus.find_substr(out[m].capitalize(), string)
            noList.append(rs)
        noList= list(chain(*noList))
        
        if len(noList)>0:
            if noList[noList.index(min(noList))]> minChar:
                string= string[0:min(noList)-1]
                file.write(string)
                file.write("\n")
            elif len(string)> noList[noList.index(max(noList))]+ minChar:
                string= string[max(noList):]
                loc= string.find(" ")+1
                string= string[loc:]
                
                if len(string)>= minChar:    
                    file.write(string)
                    file.write("\n")
        
    else:
        result.write("0")
        result.write("\n")
#        
        maxStrings= len(string)/minChar
        if maxStrings>1:
               file.write(string[0:minChar])
               file.write("\n") 
               
               string= string[minChar:]
               loc= string.find(" ")+1
               string= string[loc:]
               if len(string)>= minChar:
                   file.write(string)
                   file.write("\n")  
#            for k in range(maxStrings):
#               file.write(string[cX:cX+minChar])
#               file.write("\n")
#               cX= cX+ minChar
#        
        file.write(string)
        file.write("\n")
    if i % 100 == 0:
        print(i)
        
file.close()
result.close()