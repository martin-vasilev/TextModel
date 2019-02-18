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

D= TextDataset(txt_dir= os.getcwd() + "\\corpus\\validate.txt", 
               vocab_dir= os.getcwd() + "\\corpus\\vocab.txt",
               save_img=False, height= 210, width= 210, max_lines= 10, font_size=12, ppl=7,
               forceRGB=True, V_spacing=11, train= False, plot_grid= False,
               plot_grid_image= True)

i, wv, l, s, coords= D.__getitem__(823)

wv= wv.numpy()
l= l.numpy()
i= i.numpy()

import torch
checkpoint = torch.load('D:\Github\TextModel\VALinput.pth.tar')

imgs= checkpoint['imgs']
caps= checkpoint['caps']
caplens= checkpoint['caplens']
rawImage= checkpoint['rawImage']
coords= checkpoint['coords']
#rawImage= rawImage[0, :, :]
#a= checkpoint['imgs'].numpy()
#b= a[0,:,:,:]
#caps= checkpoint['caps'].numpy()
#caplens= checkpoint['caplens'].numpy()
#strings= checkpoint['string']

#wrong_word= []
#words= list(word_map.keys())
#for i in range(len(AllWrong)):
#    wrong_word.append(words[AllWrong[i]])
#    
#with open('wrong_words.txt', "w") as outfile:
#    for entries in wrong_word:
#        outfile.write(entries)
#        outfile.write("\n")
#        
#with open('wrong_ind.txt', "w") as outfile:
#    for entries in AllWrong_ind:
#        outfile.write(str(entries))
#        outfile.write("\n")


import os
import sys
os.chdir('D:\\Github\\TextModel')
sys.path.insert(0, './corpus')
from core.TextDataset import TextDataset


D= TextDataset(txt_dir= os.getcwd() + "\\corpus\\train.txt",
               vocab_dir= os.getcwd() + "\\corpus\\vocab.txt",
               height= 210, width= 210, max_lines= 10, font_size=12, ppl=7,
               forceRGB=True, V_spacing=11, train= False, plot_grid_image= True)

i, wv, l, img, coords= D.__getitem__(4)

wv= wv.numpy()
l= l.numpy()
i= i.numpy()