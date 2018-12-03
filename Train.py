# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:08:45 2018

@author: Martin R. Vasilev

Adapted code from: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
"""
import os
import sys
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
from core.Utils import *

if torch.cuda.is_available():
    torch.cuda.set_device(0)

# Data parameters
train_dir= '/corpus/train_test3.txt'  # location of txt file containing train strings
valid_dir= '/corpus/validate2.txt'  # location of txt file containing validate strings
token_dir = '/corpus/SUBTLEX-US.txt'  # base name shared by data files
vocab_dir = '/corpus/vocab.txt'  # base name shared by data files
data_name= 'Test_TxtModel'

# Model parameters
emb_dim = 1024#512  # dimension of word embeddings
attention_dim = 1024#512  # dimension of attention linear layers
decoder_dim = 1024#512  # dimension of the decoder RNN
dropout = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU (if available; otherwise, use CPU)
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
last_loss= 100 # to keep track of loss after last validation cycle
epochs = 1#120  # number of epochs to train for
batch_size = 1 # I run out of memory with 32 on a 6GB GPU
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
print_freq = 16  # print training/validation stats every x batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None#"checkpoint_Test_TxtModel.pth.tar"  # path to checkpoint, None if none


# load up data class:

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Load train data set:
Data= TextDataset(txt_dir= os.getcwd() + train_dir, 
               vocab_dir= os.getcwd() + vocab_dir,
               save_img=False, height= 210, width= 210,
               max_lines= 10, font_size=12, ppl=7, batch_size= 1, forceRGB=True, V_spacing=12, train= True)

word_map= Data.vocab_dict # dictionary of bocabulary and indices
Ntokens= len(word_map) # number of unique word tokens

# create separate set for validation:
ValidData= TextDataset(txt_dir= os.getcwd() + valid_dir,
               vocab_dir= os.getcwd() + vocab_dir,
               save_img=False, height= 210, width= 210,
               max_lines= 10, font_size=12, ppl=7, batch_size= 1, forceRGB=True, V_spacing=12, train= False)


def main():
    """
    Training and validation.
    """

    global checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, last_loss
    
    
    # Initialize / load checkpoint
    if checkpoint is None:
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
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        last_loss= checkpoint['last_loss']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)
    
    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Training set data loader function (pyTorch)
    train_loader = torch.utils.data.DataLoader(Data, batch_size= batch_size, shuffle= True,
                                               pin_memory= True)
    
    # Validation set data loader function (pyTorch)
    val_loader= torch.utils.data.DataLoader(ValidData, batch_size= batch_size, shuffle= True,
                                               pin_memory= True)
    
    print("Learning rate: %f\n" % (decoder_optimizer.param_groups[0]['lr'],))
    
    # Epochs
    for epoch in range(start_epoch, epochs):
        
        start_epoch= time.time()
        #adjust_learning_rate(decoder_optimizer, 4.2)
        
        # One epoch's training
        train(train_loader= train_loader,
              encoder= encoder,
              decoder= decoder,
              criterion= criterion,
              encoder_optimizer= encoder_optimizer,
              decoder_optimizer= decoder_optimizer,
              epoch= epoch)
        
        # One epoch's validation
        curr_loss= validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)
        
        
        # Did the loss go down after last epoch?
        if curr_loss< last_loss:
            # Save best one yet (so we don't overwrite)
            save_checkpoint("Test_BEST"+ data_name, epoch, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, last_loss)
            last_loss= curr_loss
        else:
            # Save checkpoint
            save_checkpoint(data_name, epoch, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, last_loss)
        end_epoch= time.time()
        print("Epoch time: %.3f minutes \n" % ((end_epoch- start_epoch)/60))
        
        
def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, string) in enumerate(train_loader):
        
#        state = {'imgs': imgs,
#             'caps': caps,
#             'caplens': caplens,
#             'string': string}
#        filename = 'checkpoint_' + data_name + '.pth.tar'
#        torch.save(state, filename)
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps= caps.to(device)
        #oneHot = oneHot.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, batch_size)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time: {batch_time.avg:.3f}\t'
                  'Data Load Time: {data_time.avg:.3f}\t'
                  'Loss: {loss.avg:.4f}\t'
                  'Accuracy: {top5.avg:.3f}'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        if encoder is not None:
            imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        top5 = accuracy(scores, targets, batch_size)
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time: {batch_time.avg:.3f}\t'
                  'Loss: {loss.avg:.4f}\t'
                  'Accuracy: {top5.avg:.3f}\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                            loss=losses, top5=top5accs))

    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}\n'.format(
            loss=losses,
            top5=top5accs))
    
    return losses.avg


if __name__ == '__main__':
    main()