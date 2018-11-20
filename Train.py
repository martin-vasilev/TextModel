# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:08:45 2018

@author: Martin R. Vasilev

Adapted code from: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
"""
import os
import sys
os.chdir('D:\\Github\\TextModel')
sys.path.insert(0, './corpus')
from core.TextDataset import TextDataset

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from core.Model import Encoder, DecoderWithAttention
#from datasets import *
#from utils import *
#from nltk.translate.bleu_score import corpus_bleu


# Data parameters
data_dir= '\\corpus\\corpus_final.txt'  # location of txt file containing train strings
token_dir = '\\corpus\\SUBTLEX-US.txt'  # base name shared by data files
Ntokens= 20000 # number of unique word tokens

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none

def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # load up data:
    Data= TextDataset(txt_dir= os.getcwd()+data_dir, corpus_dir= os.getcwd()+token_dir)
    
    # Encoder (conv net):
    encoder= Encoder()
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
                                         
    # Decoder:
    decoder = DecoderWithAttention(attention_dim= attention_dim,
                                       embed_dim= emb_dim,
                                       decoder_dim= decoder_dim,
                                       vocab_size= Ntokens,
                                       dropout= dropout)
    decoder_optimizer = torch.optim.Adam(params= filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr= decoder_lr)
    
    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    train_loader = torch.utils.data.DataLoader(
          CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
          batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    
if __name__ == '__main__':
    main()
