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
from torch.nn.utils.rnn import pack_padded_sequence

model_filename = "checkpoint_Test_BESTtry.pth.tar"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU (if available; otherwise, use CPU)

# Data parameters
test_dir= '/corpus/test.txt'  # location of txt file containing train strings
vocab_dir = '/corpus/vocab.txt'  # base name shared by data files

# Model parameters:
print_freq = 10  # print training/validation stats every x batches

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
test_loader = torch.utils.data.DataLoader(Data, batch_size= 1, shuffle= True,
                                           pin_memory= True)

# Load trained model
model = torch.load(model_filename)
decoder = model['decoder']
decoder = decoder.to(device)
decoder.eval() # evaluation mode
encoder = model['encoder']
encoder = encoder.to(device) 
encoder.eval() # evaluation mode

for i, (imgs, caps, caplens, rawImage) in enumerate(test_loader):
    
#    state = {'imgs': imgs,
#             'caps': caps,
#             'caplens': caplens,
#             'rawImage': rawImage}
#    filename = 'Testinput' + '.pth.tar'
#    torch.save(state, filename)
    
    # Move to GPU, if available
    imgs = imgs.to(device)
    caps = caps.to(device)
    caplens = caplens.to(device)
    
    # Encode the image:
    imgs = encoder(imgs)  # (1, enc_image_size, enc_image_size, encoder_dim)
    
    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
    targets = caps_sorted[:, 1:]

    # Remove timesteps that we didn't decode at, or are pads
    # pack_padded_sequence is an easy trick to do this
    scores, indx_scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
    targets, indx_targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
    
    # get image-wise matrices:
    list_targets= unflatten(targets, indx_targets, decode_lengths, False)
    list_scores= unflatten(scores, indx_scores, decode_lengths, True)
    
    # Get accuracy:
    acc, wrong, wrong_ind = accuracy(list_scores, list_targets)
    
    
    
    
    if i % print_freq == 0:
        print('Testing: [{0}/{1}]\t'.format(i, len(test_loader)))
    
    
#
#if __name__ == '__main__':
#    main()