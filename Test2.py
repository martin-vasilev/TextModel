# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:08:45 2018

@author: marti
"""
import os
import sys
os.chdir('D:\\R\\TextModel')
sys.path.insert(0, './corpus')

from TextDataset import TextDataset

D= TextDataset(txt_dir= os.getcwd() + "\\corpus\\corpus_final.txt",
               corpus_dir= os.getcwd() + "\\corpus\\SUBTLEX-US.txt")

s= D.GetText()