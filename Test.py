# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:14:21 2018

@author: mvasilev
"""

import os
os.chdir('C:\\Users\\mvasilev\\TextModel')

# load txt data:
with open(os.getcwd() + '\corpus\sample.txt', 'r') as myfile:
        data= myfile.read()
text= data.split('\n')

from textGenerator import textGenerator

text_list, image_list= textGenerator(text, save_img=True)