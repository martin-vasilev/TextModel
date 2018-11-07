# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:14:21 2018

@author: mvasilev
"""

import os
import sys

os.chdir('D:\\R\\TextModel')
sys.path.insert(0, './corpus')

# load txt data:
with open(os.getcwd() + '\corpus\corpus_final.txt', 'r') as myfile:
        data= myfile.read()
text= data.split('\n')

from textGenerator import textGenerator
from Corpus import Corpus

text_list, images= textGenerator(text, save_img=True, max_lines= 6, batch_size= 20)

### Random words:

tokens= Corpus.SUBTLEX(20000) # get N SUBTLEX tokens
text_list, images= textGenerator(text, input_method= "words", save_img=True, max_lines= 6, batch_size= 20)
