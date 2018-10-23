# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:00:34 2018

@author: mvasilev
"""

# courier: 14 char: 8ppl

def textGenerator(text, input_method= "words", batch_size=5, height=140, width= 600, noise= 0, words_per_line= 12, max_lines= 4,
                   font= "cour", font_size= 14, ppl=8, save_img= False):
    
    import random
#    import math
    import numpy as np
    
    images = np.zeros((batch_size, height,width))
    text_list= []
    
    # take random text strings:
    batch_texts= random.sample(text, batch_size)
    
    # Generate text strings that will be used in the batch:
    for i in range(batch_size): # for each element in batch size..
#        MaxNwords=  words_per_line*max_lines # max number of words given input constraints
#        useText= batch_texts[i]
#        word_list= useText.split(' ')
#        
#        if len(word_list)> MaxNwords: # get only words we need (if text is longer)
#            useWords= word_list[0:MaxNwords]
#        else: # otherwise take what's available
#            useWords= word_list
#        
#        # Parse text into lines:
#        # possible # of lines given input (necessary if text is smaller than what is needed):
#        actualNlines= int(math.ceil(len(useWords)/words_per_line))
#        
#        # create string of text to be used:
#        start= 0
#        string= ""
#        for j in range(actualNlines):
#            if j== actualNlines:
#                line= [useWords[k] for k in range(start, len(useWords)-1)]
#            else:
#                line= [useWords[k] for k in range(start, start+words_per_line)]
#            
#            line_string= ' '.join(line)
#            
#            if j>0: # if not first line:
#                string= string+ "\n"+ line_string
#            else:
#                string= line_string # 1st line
#            start= start +words_per_line
        useWords= batch_texts[i].split(' ')
        textDone= False
        currPos= 10 # starting x value of print function
        w= 0
        line=1
        string= ""
        while not textDone:
            if currPos+ (len(useWords[w])+1)*ppl < width-10: # if text still fits on current line..
                if w>0:
                    string= string + " "+ useWords[w]
                    currPos= currPos+ len(" "+ useWords[w])*ppl
                else:
                    string= string + useWords[w]
                    currPos= currPos+ len(useWords[w])*ppl
                
            else: # therwise move on next line..
                line= line+1
                if line> max_lines:
                    textDone= True
                else:
                    currPos= 10 + len(useWords[w])*ppl
                    string= string + "\n"+ useWords[w] # break line
            
            #textDone= line== max_lines and currPos+ (len(useWords[w])+1)*ppl > width-20
            w= w+1 # go to next word
            if w== len(useWords): # no more text to use, stop
                textDone= True
        
        text_list.append(string)
        
        ############
        # Generate images using the text:
        from PIL import Image, ImageDraw, ImageFont#, ImageFilter
        #whichFont= font+ '.ttf'
        font = ImageFont.truetype('Fonts/cour.ttf', font_size)
        
        #img = Image.new('L', (width, height), color = 'white') # open an empty canvas
        # https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes
        
        # open a random canvas image as background:
        img= Image.open("canvas/" + str(np.random.randint(1, 10))+ ".jpg").convert('L')
        
        # crop canvas to desired size:
        canvWidth, canvHeight= img.size # get canvas size
        
        # get a random square from the canvas to crop:
        done= False
        xDone= False
        yDone= False

        while not done:
            xGuess= np.random.randint(1, canvWidth) # guess x start value
            yGuess= np.random.randint(1, canvHeight) # guess y start value
            
            if (xGuess + width) <= canvWidth:
                x1= xGuess
                x2= xGuess+ width
                xDone= True
            
            if (yGuess+ height) <= canvHeight:
                y1= yGuess
                y2= yGuess + height
                yDone= True
                
            done= yDone and xDone # selection is ok if there is enough space to cut
        
        img= img.crop(box= (x1, y1, x2, y2)) # crop canvas to fit image size
        
        d = ImageDraw.Draw(img) # draw canvas
        d.multiline_text((10,10), string, font= font, spacing= 20, align= "left") # draw text
        
        # add compression/decompression variability to the image:
        img.save("template.jpeg", "JPEG", quality=np.random.randint(30, 100))
        img= Image.open("template.jpeg").convert('L')
        
        img= (np.array(img)) # convert to numpy array
        
#        ### Add noise:
#        from scipy.ndimage import gaussian_filter
#        img = gaussian_filter(img, sigma= noise)
        
        images[i, :, :]= img # add current image to batch image array
        
        if save_img:
            if noise != 0:
                filename= 'img' + str(i+1)+ "_N"+ '.png'
            else:
                filename= 'img' + str(i+1)+ '.png'
            from scipy import misc
            misc.imsave(filename, img)
            #im = Image.fromarray(img)
            #im.save('img' + str(i+1)+ '.png')
        
    return text_list, images