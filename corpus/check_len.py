# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:39:25 2018

@author: mvasilev
"""

import os
#import numpy as np
from Corpus import Corpus
#from itertools import chain
os.chdir('D:\\COCA\\preproc')

minChar= 60*6
file = open("corpus_acad.txt", "w")

with open("good_checked_acad.txt", 'r') as myfile:
    data= myfile.read()
    text= data.split('\n')
text = list(filter(None, text))
text= Corpus.unique_list(text)

a= 1
for i in range(len(text)):
    string= text[i]
    
    if len(string)>= 2*minChar:
            file.write(string[0:minChar]) # 1st part
            file.write("\n")
            string= string[minChar:]
            loc= string.find(" ")+1
            string= string[loc:]
            if len(string)>= minChar:
                print(a)
                a= a+1
                file.write(string) # 2nd part
                file.write("\n")
    else:
        file.write(string)
        file.write("\n")
        
file.close()