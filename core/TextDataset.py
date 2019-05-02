# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 11:43:35 2018

@author: Martin R. Vasilev
"""
from torch.utils.data import Dataset
from core.Corpus import Corpus

class TextDataset(Dataset):
    """Text string dataset."""

    def __init__(self, txt_dir, vocab_dir, input_method= "text", batch_size=1, height=120,
                 width= 480, max_lines= 6, font_size= 14, ppl=8, V_spacing= 15, uppercase= False,
                 save_img= False, forceRGB= False, transform=None, max_words= 170, train= True,
                 plot_grid= False, plot_grid_image= False): # 112 max words
        """
        Input:
            txt_dir:      Path to the text corpus file containing the input strings.
            root_dir:     Directory with the SUBTLEX-US corpus file (used for getting dictionary).
            vocab_dir:    Directory of vocabulary txt file
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
            forceRGB      Make it output an RGB (3-channel) image (used for testing/ development)
			transform	  Image transformation (if any)
            plot_grid     A logical indicating whether to plot text grid lines (vertical ones)
            plot_grid_image   A logical indicating whether to save a version of the image with grids for each word token (validation)
        """
        # load txt data:
        with open(txt_dir, 'r') as myfile:
            data= myfile.read()
        self.text= data.split('\n')
        
        with open(vocab_dir, 'r') as myfile:
            data= myfile.read()
        self.vocab= data.split('\n')    
        #self.vocab= Corpus.SUBTLEX(N, corpus_dir) # get N SUBTLEX tokens
        
        # Other parameters:
        self.vocab_size= len(self.vocab)
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
        self.forceRGB= forceRGB
        self.max_words= max_words
        self.train= train
        self.pix_per_line= self.height// self.max_lines
        self.plot_grid= plot_grid
        self.plot_grid_image= plot_grid_image
        
		# PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        
        # word dictionary (from vocab:
        vocab_dict = {}

        for i in range(self.vocab_size):
            vocab_dict[self.vocab[i]]= i
        
        self.vocab_dict= vocab_dict
        self.vocab_size= len(vocab_dict)

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
        import os
        import sys
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont#, ImageFilter
        from scipy import misc
        import torch
        
        images = np.zeros((self.batch_size, self.height, self.width))
        #oneHot = np.zeros((self.vocab_size, self.batch_size))
        text_list= []
        
        # take random text strings:
        if self.input_method== "text":
            if item is None: # take random sample if item number not provided
                item= np.random.randint(0, len(self.text)-1)
                batch_texts= self.text[item]
            else:
                batch_texts= self.text[item]
        elif self.input_method== "words": # random word input method
            self.extra_tokens= {".": 0.050197381848061426, ",": 0.047120642142730866, "!": 0.0008645381544422652, "?": 0.00429955163340679, "(": 0.0011715728854572677, ")": 0.0010669640018258278, ":": 0.0036394702435669033, ";": 0.0008602099702054456, "\\": 0.016251299703785728, "#": 0.0007539696940539737, "%": 8.163621345139732e-05, "<num>": 0.0143638784392834}
            batch_texts= []
            words= random.sample(self.vocab[1:19536], self.batch_size*100) # take 120 random words per batch to be safe- we discard the rest later
            # generate randomly special tokens (e.g., numbers, punctuations marks) using real text probabilities:
            for tkn in self.extra_tokens:
                # draw random numbers for curr extra token:
                rnd= np.random.uniform(0, 1, 100)
                hits= rnd[rnd<= self.extra_tokens[tkn]] # how many draws meet prob. criterion?
                # add number of hits to exisiting tokens
                if tkn== "<num>":
                    rnd_num= np.random.randint(0, 9, len(hits))
                    rnd_num= list(map(str, rnd_num))
                    words.extend(rnd_num)
                else:
                    words.extend([tkn]*len(hits))
            
            # shuffle new tokens so that they are not always at the end:
            random.shuffle(words)
            batch_texts= " ".join(words) # make a string
#            for k in range(self.batch_size):
#                string= " ".join(words[0:120])
#                batch_texts.append(string)
#                del words[0:120] # remove selection from the remaining words
        else:
            sys.exit("Input method not supported!")
        
        # Generate text strings that will be used in the batch:
        for i in range(self.batch_size): # for each element in batch size..
            useWords= batch_texts.split(' ')
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
            d.multiline_text((1,6), string, font= font, spacing= self.V_spacing, align= "left") # draw text
            
            # plot grid lines?
            if self.plot_grid:
                # do something
                coords = list(range(0, self.height, self.pix_per_line))
                for l in range(len(coords)):
                    d.line([0, coords[l], self.width, coords[l]], fill= 'green')
                    d.line([coords[l], 0, coords[l], self.height], fill= 'green')
            # add compression/decompression variability to the image:
            filename= "temp" + str(item) + ".jpeg"
            img.save(filename, "JPEG", quality=np.random.randint(30, 100))
            img= Image.open(filename).convert('L')
            os.remove(filename)
            
            img= (np.array(img)) # convert to numpy array            
            rawImage= img # keep a copy of the image (for testing)
            
            if self.batch_size>1:
                images[i, :, :]= img # add current image to batch image array
            else:
                images= img
                if self.forceRGB: # make fake 3-channel image (for testing)
                    img_n= np.zeros((3, self.height, self.width))
                    img_n[0,:,:]= images
                    img_n[1,:,:]= images
                    img_n[2,:,:]= images
                    images= img_n
                
            if self.save_img:
                filename= 'img' + str(item)+ '.png'
                misc.imsave(filename, img)
                
            # Turn text into a one-hot vector:
            string2= string.replace('\n', ' ')
            wrds= Corpus.strip2(string2)
            
            word_vec= np.zeros(self.max_words +2)
            word_vec[0]= self.vocab_dict['<start>']
            numbers= ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            
            for t in range(len(wrds)):
                if wrds[t] in self.vocab:
                    word_vec[t+1]= self.vocab_dict[wrds[t]]
                else:
                    if wrds[t] in numbers:
                        word_vec[t+1]= self.vocab_dict['<num>'] # code all numbers as "num"
                    else:
                        word_vec[t+1]= self.vocab_dict['<unk>']
                    #print("'"+wrds[t]+ "'"+ " not found in dictionary") 
            
            word_vec[len(wrds)+1]= self.vocab_dict['<end>']                
        
        # convert to torch tensors:
        images= torch.FloatTensor(images/ 255.)
        word_vec= torch.LongTensor(word_vec)
        
        if not self.train: # get grid lines for token boxes (i.e., pixel coordinates for token location on the image)
            V_spacing= self.V_spacing-1
            V_ppl= 11 # vertical height of text
            punct= [".", ",", "!", "?", "(", ")", ":", ";", "%"] # '"', "#"
            numbers= ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            
            lines= string.split('\n')
            AllTkn= Corpus.strip2(string.replace("\n", " ")) # all tokens
            
            coords= []
            currY= 6
            
            for i in range(len(lines)):
                words= Corpus.strip2(lines[i]) # get word tokens
                currX= 1 # reset x on each line
                
                if i>0: # increment line y coords
                    currY= currY+ V_spacing+ V_ppl
                
                for j in range(len(words)):
                    startX= currX # start pixel of word token
                    endX= currX+ len(words[j])*self.ppl
                
                    currX= endX+self.ppl # add empty space
                    
                    # special cases- abbrev:
                    if words[j]== "have" and "'ve" in lines[i]:
                        startX= startX- self.ppl
                        endX= endX - 2*self.ppl # -2 bc 1 letter is abbreviated + appostrophe
                        currX= currX -2*self.ppl
                        
                    if words[j]== "are" and "'re" in lines[i]:
                        startX= startX- self.ppl
                        endX= endX - self.ppl 
                        currX= currX - self.ppl
                        
                    if words[j]== "will" and "'ll" in lines[i]:
                        startX= startX- self.ppl
                        endX= endX - 2* self.ppl # -2 bc 1 letter is abbreviated + appostrophe
                        currX= currX -2* self.ppl
                        
                    if words[j]== "would" and "'d" in lines[i]:
                        startX= startX- 3* self.ppl
                        endX= endX - 4* self.ppl 
                        currX= currX -4* self.ppl
                        
                    ## 
                    if words[j]== "not" and "n't" in lines[i]:
                        startX= startX- self.ppl
                        endX= endX - self.ppl
                        currX= currX - self.ppl
                    
                    if words[j]== "am" and "i'm" in lines[i]:
                        startX= startX- self.ppl
                        endX= endX - self.ppl
                        currX= currX - self.ppl
                        
                    if words[j] in punct:
                        startX= startX- self.ppl
                        endX= endX- self.ppl
                        currX= currX- self.ppl
                    if j< len(words)-1:
                        if words[j] in numbers and words[j+1] in numbers:
                           currX= currX- self.ppl 
                    
                    word_coords= (startX, currY-2, endX, currY+ V_ppl+2) # save coords
                    coords.append(word_coords)
            
            if len(AllTkn) != len(coords):
                coords= coords[:len(AllTkn)]
                
            # output as matrix for easier handling:
            dim_mat= np.zeros((self.max_words +2, 1, 4)) # here we pad to the same size because number of dims differs across images

            for d in range(len(coords)):
                dim_mat[d,]= coords[d]       
            
            if self.plot_grid_image:
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                
                fig, ax = plt.subplots()
                fig.set_tight_layout(False)
                ax.tick_params(axis='both', left='off', top='off', right='off', bottom='on',
                                   labelleft='off', labeltop='off', labelright='off', labelbottom='off')
                ax.imshow(rawImage, cmap='Greys_r')
                
                for k in range(len(coords)):
                    rect = patches.Rectangle((coords[k][0], coords[k][1]), coords[k][2]- coords[k][0], coords[k][3]- coords[k][1],
                                             linewidth=1,edgecolor='r',facecolor='none')
                    ax.add_patch(rect)
                #plt.show()
                plt.savefig('grids_' + str(item)+ '.png')

        ###### transform:
        if self.transform is not None:
            images = self.transform(images)
        if self.train:
            return images, word_vec, torch.LongTensor([len(wrds)+2]), string
        else:
            
            return images, word_vec, torch.LongTensor([len(wrds)+2]), rawImage, torch.LongTensor(dim_mat)
            