# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 11:43:35 2018

@author: Martin R. Vasilev
"""
from torch.utils.data import Dataset
from core.Corpus import Corpus

class TextDataset(Dataset):
    """Text string dataset."""

    def __init__(self, txt_dir, corpus_dir, N= 20000, input_method= "text", batch_size=1, height=120,
                 width= 480, max_lines= 6, font_size= 14, ppl=8, V_spacing= 7, uppercase= False,
                 save_img= False):
        """
        Input:
            txt_dir:      Path to the text corpus file containing the input strings.
            root_dir:     Directory with the SUBTLEX-US corpus file (used for getting dictionary).
            N:            number of tokens to be used in the dictionary.
                          input_method: a character denoting what type of text to use. Default ('text') just used natural 
                          language text taken from the corpus. Alternatively, 'words' creates a "text" 
                          consisting of randomly-generated words (taken from the dictionary, i.e., SUBTLEX-US).
            batch_size:   number of images to use in a batch
            height:       height of the text image
            width:        width of the text image
            max_lines:    number of lines of text in the image. Note that the model has been developed with the
                          default number of lines. Changing this may alter model performance
            font_size:    size of the font to be use (default font is Courier). If you change the font of the text,
                          be sure to used a fixed-wdith (i.e., monospaced) alternative.
            ppl:          pixels per letter. This denoes the width (in pixels) of each letter. Because we use a 
                          monospaced font, this number is a constant.
                          14:8, 12:7,
            V_spacing     Vertical spacing of the text (i.e., how many blank pixels are between lines)  
            uppercase     A logical indicating whether to format the text uppercase or not
            save_img      A logical indicating whether to save the images locally (for testing)
        """
        # load txt data:
        with open(txt_dir, 'r') as myfile:
            data= myfile.read()
        self.text= data.split('\n')
        
        self.tokens= Corpus.SUBTLEX(N, corpus_dir) # get N SUBTLEX tokens
        self.Ntokens= len(self.tokens)
        self.input_method= input_method
        self.batch_size= batch_size
        self.height= height
        self.width= width
        self.max_lines= max_lines
        self.font_size= font_size
        self.ppl= ppl
        self.V_spacing= V_spacing
        self.uppercase= uppercase
        self.save_img= save_img

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item= None):
        
        """
        Generates input to be used in the model training. It outputs a matrix (N_batch, H, W) containing 
        the text images and a list containing the character strings. Note that these are grayscale images, so
        they always have only 1 channel.
        
        Input:
            item:         index of the item to be used for generating the output. If None, a random item is taken.
            
        Output:
                          Generated image, text string, and one-hot vector
        """
    
        import random
        import sys
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont#, ImageFilter
        from scipy import misc
        
        images = np.zeros((self.batch_size, self.height, self.width))
        oneHot = np.zeros((self.Ntokens, self.batch_size))
        text_list= []
        
        # take random text strings:
        if self.input_method== "text":
            if item is None: # take random sample if item number not provided
                batch_texts= random.sample(self.text, self.batch_size)
            else:
                batch_texts= self.text[item]
        elif self.input_method== "words": # random word input method
            batch_texts= []
            words= random.sample(self.tokens, self.batch_size*120) # take 120 random words per batch to be safe- we discard the rest later
            for k in range(self.batch_size):
                string= " ".join(words[0:120])
                batch_texts.append(string)
                del words[0:120] # remove selection from the remaining words
        else:
            sys.exit("Input method not supported!")
        
        # Generate text strings that will be used in the batch:
        for i in range(self.batch_size): # for each element in batch size..
    
            useWords= batch_texts[i].split(' ')
            textDone= False
            currPos= 1 # starting x value of print function
            w= 0
            line=1
            string= ""
            while not textDone:
                if currPos+ (len(useWords[w])+1)*self.ppl < self.width-10: # if text still fits on current line..
                    if w>0:
                        string= string + " "+ useWords[w]
                        currPos= currPos+ len(" "+ useWords[w])*self.ppl
                    else:
                        string= string + useWords[w]
                        currPos= currPos+ len(useWords[w])*self.ppl
                    
                else: # therwise move on next line..
                    line= line+1
                    if line> self.max_lines:
                        textDone= True
                    else:
                        currPos= 1 + len(useWords[w])*self.ppl
                        string= string + "\n"+ useWords[w] # break line
                
                #textDone= line== max_lines and currPos+ (len(useWords[w])+1)*ppl > width-20
                w= w+1 # go to next word
                if w== len(useWords): # no more text to use, stop
                    textDone= True
            if self.uppercase:        
                string= string.upper() # make string upper case
            else:
                string= string.lower() # make string lower case
            text_list.append(string) # add to text list
            
            ############
            # Generate images using the text:
            font = ImageFont.truetype('Fonts/cour.ttf', self.font_size)
            
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
                
                if (xGuess + self.width) <= canvWidth:
                    x1= xGuess
                    x2= xGuess+ self.width
                    xDone= True
                
                if (yGuess+ self.height) <= canvHeight:
                    y1= yGuess
                    y2= yGuess + self.height
                    yDone= True
                    
                done= yDone and xDone # selection is ok if there is enough space to cut
            
            img= img.crop(box= (x1, y1, x2, y2)) # crop canvas to fit image size
            
            d = ImageDraw.Draw(img) # draw canvas
            d.multiline_text((1,1), string, font= font, spacing= self.V_spacing, align= "left") # draw text
            
            # add compression/decompression variability to the image:
            img.save("template.jpeg", "JPEG", quality=np.random.randint(30, 100))
            img= Image.open("template.jpeg").convert('L')
            
            img= (np.array(img)) # convert to numpy array            
            images[i, :, :]= img # add current image to batch image array
            
            if self.save_img:
                filename= 'img' + str(i+1)+ '.png'
                misc.imsave(filename, img)
                
            # Turn text into a one-hot vector:
            string= string.replace('\n', ' ')
#            string= string.replace('-', ' ')
#            string= string.replace('/', '')
#            string= string.replace("\\", ' ')
            wrds= Corpus.strip(string)
            
            for t in range(len(wrds)):
                if wrds[t] in self.tokens:
                    ind= self.tokens.index(wrds[t])
                    oneHot[ind,i]= 1
                else:
                    print(wrds[t]+ " not found in dictionary")                    
               
        #sample= {'images': images, 'text': text_list, 'oneHot': oneHot}
        return images, text_list, oneHot