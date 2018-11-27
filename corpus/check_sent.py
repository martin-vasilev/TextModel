# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:30:33 2018

@author: Martin R. Vasilev
"""
import os
#import numpy as np
from Corpus import Corpus
#from itertools import chain
os.chdir('D:\\COCA\\preproc')

minChar= 60*6
c= ['.', '!', '?']

file = open("news_final.txt", "w")

with open("good_checked_news.txt", 'r') as myfile:
    data= myfile.read()
    text= data.split('\n')
text = list(filter(None, text))
text= Corpus.unique_list(text)

for i in range(len(text)):
    string= text[i]
    
    if string[0]== " ":
        string= string[1:]
        
    # specifically for spoken corpus:
    if " @!" in string:
        pos= Corpus.find_substr(" @!", string)
        for k in range(len(pos)):
            if string[pos[k]-1] in c:
                string= string[0:pos[k]+1] + "Ö"*2 + string[pos[k]+3:]
            else:
                string= string[0:pos[k]] + '. ' + "Ö"*1 + string[pos[k]+3:]
        string= string.replace("Ö", "")
    
    
    # check if we start with a new sent:
    if string[0].islower():
        nPos= [pos for pos, char in enumerate(string) if char in c]
        if not nPos:
            continue
        nPos= min(nPos)
        string= string[nPos+1:]
        if len(string)==0:
            continue
        if string[0]== " ":
            string= string[1:]
    
    if string[len(string)-1] not in c:
        nPos= [pos for pos, char in enumerate(string) if char in c]
        if not nPos:
            continue
        nPos= max(nPos)
        string= string[:nPos+1]
        if len(string)==0:
            continue
        
    if len(string)> minChar: # cut strings that would be too long to fit our image:
        string= string[:minChar-1]
        if string[len(string)-1] not in c:
            nPos= [pos for pos, char in enumerate(string) if char in c]
            if not nPos:
                continue
            nPos= max(nPos)
            string= string[:nPos+1]
        if len(string)==0:
            continue
        
    if len(string)>= minChar - 2*60:
        file.write(string)
        file.write("\n")
        
file.close()