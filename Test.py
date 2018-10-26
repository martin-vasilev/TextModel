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

text_list, images= textGenerator(text, save_img=True, max_lines= 6, batch_size= 20)


#useWords= batch_texts[i].split(' ')
#textDone= False
#currPos= 10 # starting x value of print function
#w= 0
#line=1
#string= ""
#while not textDone:
#    if currPos+ (len(useWords[w])+1)*ppl < width-20: # if text still fits on current line..
#        if w>0:
#            string= string + " "+ useWords[w]
#            currPos= currPos+ len(" "+ useWords[w])*ppl
#        else:
#            string= string + useWords[w]
#            currPos= currPos+ len(useWords[w])*ppl
#        
#    else: # therwise move on next line..
#        line= line+1
#        currPos= 10
#        string= string + "\n"+ useWords[w] # break line
#    
#    textDone= line== max_lines and currPos+ (len(useWords[w])+1)*ppl > width-20
#    w= w+1 # go to next word
##

# useful stuff:
# http://www.scipy-lectures.org/advanced/image_processing/

#from scipy import misc
#from scipy.ndimage import gaussian_filter
#
#img= images[0]
#img = gaussian_filter(img, sigma=1.2)
#misc.imsave('test2.png', img)
#
#
#nums = numpy.random.choice([0, 1], size= (height, width), p=[.1, .9])
#img2= img.convert("RGB")
#