# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:08:45 2018

@author: marti
"""
import os
import sys
os.chdir('D:\\Github\\TextModel')
sys.path.insert(0, './corpus')
from core.TextDataset import TextDataset

D= TextDataset(txt_dir= os.getcwd() + "\\corpus\\corpus_final.txt", 
               corpus_dir= os.getcwd() + "\\corpus\\SUBTLEX-US.txt",
               vocab_dir= os.getcwd() + "\\corpus\\vocab.txt",
               save_img=False, height= 210, width= 210,
               max_lines= 12, font_size=12, ppl=7, batch_size= 1, forceRGB=True)

i, wv, l, s= D.__getitem__()

wv= wv.numpy()
l= l.numpy()
i= i.numpy()

import torch
checkpoint = torch.load('D:\Github\TextModel\input.pth.tar')

a= checkpoint['imgs'].numpy()
b= a[0,:,:,:]
caps= checkpoint['caps'].numpy()
caplens= checkpoint['caplens'].numpy()
strings= checkpoint['string']
