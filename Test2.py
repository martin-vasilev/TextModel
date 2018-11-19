# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:08:45 2018

@author: marti
"""
import os
import sys
os.chdir('D:\\Github\\TextModel')
sys.path.insert(0, './corpus')
import torch
from core.TextDataset import TextDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

D= TextDataset(txt_dir= os.getcwd() + "\\corpus\\corpus_final.txt",
               corpus_dir= os.getcwd() + "\\corpus\\SUBTLEX-US.txt")

s= D.GetText(save_img=False, height= 210, width= 210, max_lines= 12, font_size=12, ppl=7)