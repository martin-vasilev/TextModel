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
               save_img=True, height= 210, width= 210, max_lines= 10, font_size=12, ppl=7,
               forceRGB=True, V_spacing=11, train= False, plot_grid= False,
               plot_grid_image= False, input_method= "words")

i, wv, l, s, coords= D.__getitem__(76)

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

#########################################################################################
import os
import sys
os.chdir('D:\\Github\\TextModel')

valid_dir= '/corpus/validate.txt'  # location of txt file containing validate strings
vocab_dir = '/corpus/vocab.txt'  # base name shared by data files

ValidData= TextDataset(txt_dir= os.getcwd() + valid_dir,
               vocab_dir= os.getcwd() + vocab_dir,
               height= 210, width= 210, max_lines= 10, font_size=12, ppl=7,
               forceRGB=True, V_spacing=11, train= False)# save_img= True, plot_grid= True)

word_map= ValidData.vocab_dict # dictionary of vocabulary and indices
words= list(word_map.keys())

Alltokens= []
tokens= []

for i in range(1000):
    img, wv, l, img, coords= ValidData.__getitem__(i)
    wv= wv.numpy()
    wv= wv[wv>0]
    
    for j in range(len(wv)):
        tokens.append(words[wv[j]])
    Alltokens.extend(tokens)
    
    print(i)
    
# create dict:
punct= {}

for i in range(13):
    item= 19537+i
    punct[vocab[item]]= Alltokens.count(vocab[item])/len(Alltokens)

a= {".": 0.050197381848061426, ",": 0.047120642142730866, "!": 0.0008645381544422652, "?": 0.00429955163340679, "(": 0.0011715728854572677, ")": 0.0010669640018258278, ":": 0.0036394702435669033, ";": 0.0008602099702054456, "\\": 0.016251299703785728, "#": 0.0007539696940539737, "%": 8.163621345139732e-05, "<num>": 0.0143638784392834, "<unk>": 0.0008132658180984023}