# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 10:18:28 2018

@author: Martin R. Vasilev, 2018
"""
# Tests the trained Reading model:
import os
from core.TextDataset import TextDataset

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from core.Utils import *
import torch.nn.functional as F
from tqdm import tqdm

model_filename = "checkpoint_Test_BESTTest_TxtModel.pth.tar"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU (if available; otherwise, use CPU)

# Data parameters
test_dir= '/corpus/test.txt'  # location of txt file containing train strings
vocab_dir = '/corpus/vocab.txt'  # base name shared by data files


#def main():

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Load Test data set:
Data= TextDataset(txt_dir= os.getcwd() + test_dir, 
               vocab_dir= os.getcwd() + vocab_dir,
               save_img=False, height= 210, width= 210,
               max_lines= 10, font_size=12, ppl=7, batch_size= 1, forceRGB=True, V_spacing=12, train= False)

word_map= Data.vocab_dict # dictionary of vocabulary and indices
Ntokens= len(word_map) # number of unique word tokens


# Test set data loader function (pyTorch)
# Here, we use batch_size= 1
train_loader = torch.utils.data.DataLoader(Data, batch_size= 1, shuffle= True,
                                           pin_memory= True)

# Load trained model
model = torch.load(model_filename)
decoder = model['decoder']
decoder = decoder.to(device)
decoder.eval() # evaluation mode
encoder = model['encoder']
encoder = encoder.to(device) # evaluation mode
encoder.eval()

#
#if __name__ == '__main__':
#    main()