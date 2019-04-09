# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 23:57:33 2018

@author: Martin R. Vasilev
"""
import os
os.chdir('D:\\COCA\\preproc')

alltxt= []

with open("spok_final.txt", 'r') as myfile:
    data= myfile.read()
    text= data.split('\n')

alltxt.extend(text)

with open("acad_final.txt", 'r') as myfile:
    data= myfile.read()
    text= data.split('\n')
    
alltxt.extend(text)

with open("mag_final.txt", 'r') as myfile:
    data= myfile.read()
    text= data.split('\n')
    
alltxt.extend(text)

with open("fict_final.txt", 'r') as myfile:
    data= myfile.read()
    text= data.split('\n')
    
alltxt.extend(text)

with open("news_final.txt", 'r') as myfile:
    data= myfile.read()
    text= data.split('\n')
    
alltxt.extend(text)


# shuffle strings:
import random
random.shuffle(alltxt)

# division: 100 000 for trainng, 1 000 for validation, 3608 for test

train= alltxt[:100000]
validate= alltxt[100000:101000]
test= alltxt[101000:]

outF = open("D:\\Github\\TextModel\\corpus\\all_corpus.txt", "w")
for i in range(len(alltxt)):
    outF.write(alltxt[i])
    outF.write("\n")
outF.close()

outF = open("D:\\Github\\TextModel\\corpus\\train.txt", "w")
for i in range(len(train)):
    outF.write(train[i])
    outF.write("\n")
outF.close()

outF = open("D:\\Github\\TextModel\\corpus\\validate.txt", "w")
for i in range(len(validate)):
    outF.write(validate[i])
    outF.write("\n")
outF.close()

outF = open("D:\\Github\\TextModel\\corpus\\test.txt", "w")
for i in range(len(test)):
    outF.write(test[i])
    outF.write("\n")
outF.close()

