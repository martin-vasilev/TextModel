# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:00:34 2018

@author: mvasilev
"""

# courier: 14 char: 8ppl

def textGenerator(text, input_method= "text", batch_size=5, height=120, width= 480, max_lines= 6,
                  font_size= 14, ppl=8, save_img= False):
    
    import random
    import sys 
#    import math
    import numpy as np
    
    images = np.zeros((batch_size, height,width))
    text_list= []
    
    # take random text strings:
    
    if input_method== "text":
        batch_texts= random.sample(text, batch_size)
    elif input_method== "words": # random word input method
        batch_texts= []
        words= random.sample(text, batch_size*120) # take 120 random words per batch to be safe- we discard the rest later
        for k in range(batch_size):
            string= " ".join(words[0:120])
            batch_texts.append(string)
            del words[0:120] # remove selection from the remaining words
    else:
        sys.exit("Input method not supported!")
    
    # Generate text strings that will be used in the batch:
    for i in range(batch_size): # for each element in batch size..

        useWords= batch_texts[i].split(' ')
        textDone= False
        currPos= 1 # starting x value of print function
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
                    currPos= 1 + len(useWords[w])*ppl
                    string= string + "\n"+ useWords[w] # break line
            
            #textDone= line== max_lines and currPos+ (len(useWords[w])+1)*ppl > width-20
            w= w+1 # go to next word
            if w== len(useWords): # no more text to use, stop
                textDone= True
        string= string.lower() # make string lower case
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
        d.multiline_text((1,1), string, font= font, spacing= 8, align= "left") # draw text
        
        # add compression/decompression variability to the image:
        img.save("template.jpeg", "JPEG", quality=np.random.randint(30, 100))
        img= Image.open("template.jpeg").convert('L')
        
        img= (np.array(img)) # convert to numpy array
        
        images[i, :, :]= img # add current image to batch image array
        
        if save_img:
            filename= 'img' + str(i+1)+ '.png'
            from scipy import misc
            misc.imsave(filename, img)

    return text_list, images