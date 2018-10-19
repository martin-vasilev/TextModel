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

text_list, images= textGenerator(text, save_img=True, max_lines= 7, noise= 0)

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
