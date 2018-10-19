# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:00:34 2018

@author: mvasilev
"""

def textGenerator(text, batch_size=5, height=300, width= 600, noise= 0, words_per_line= 12, max_lines= 8,
                   font= "arial", font_size= 12, save_img= False):
    
    import random
    import math
    import numpy as np
    
    images = np.zeros((batch_size, height,width))
    text_list= []
    
    # take random text strings:
    batch_texts= random.sample(text, batch_size)
    
    
    # Generate text strings that will be used in the batch:
    for i in range(batch_size): # for each element in batch size..
        MaxNwords=  words_per_line*max_lines # max number of words given input constraints
        useText= batch_texts[i]
        word_list= useText.split(' ')
        
        if len(word_list)> MaxNwords: # get only words we need (if text is longer)
            useWords= word_list[0:MaxNwords]
        else: # otherwise take what's available
            useWords= word_list
        
        # Parse text into lines:
        # possible # of lines given input (necessary if text is smaller than what is needed):
        actualNlines= int(math.ceil(len(useWords)/words_per_line))
        
        # create string of text to be used:
        start= 0
        string= ""
        for j in range(actualNlines):
            if j== actualNlines:
                line= [useWords[k] for k in range(start, len(useWords)-1)]
            else:
                line= [useWords[k] for k in range(start, start+words_per_line)]
            
            line_string= ' '.join(line)
            
            if j>0: # if not first line:
                string= string+ "\n"+ line_string
            else:
                string= line_string # 1st line
            start= start +words_per_line
        
        text_list.append(string)
        
        ############
        # Generate images using the text:
        from PIL import Image, ImageDraw, ImageFont
        #whichFont= font+ '.ttf'
        font = ImageFont.truetype('arial.ttf', font_size)
        
        img = Image.new('1', (width, height), color = 'white') # open image
        d = ImageDraw.Draw(img) # draw canvas
        d.multiline_text((10,10), string, font= font, spacing= 20, align= "left") # draw text
        
        if save_img:
            img.save('img' + str(i+1)+ '.png')
        
        img= (np.array(img)) # convert to numpy array
        images[i, :, :]= img # add current image to batch image array
        
    return text_list, images