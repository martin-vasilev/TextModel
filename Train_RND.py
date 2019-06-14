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

import pandas as pd
import numpy as np
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from core.Model import Encoder, DecoderWithAttention
from core.Utils import *
from operator import itemgetter
from PIL import Image


# Data parameters & settings
train_dir= '/corpus/train.txt'  # location of txt file containing train strings
valid_dir= '/corpus/validate.txt'  # location of txt file containing validate strings
vocab_dir = '/corpus/vocab.txt'  # base name shared by data files
data_name= '10x10RND'
result_filename= 'Val_resultsRND'
TrainModel= True # set to false for validation round only
save_worst_image= False # save the worst image (in terms of accuracy) for later inspection/ testing
save_animation= False # save a gif animation of the model performance


# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of the decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU (if available; otherwise, use CPU)
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 3  # number of epochs to train for
batch_size = 16 # I run out of memory with 32 on a 6GB GPU
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
print_freq = 20  # print training/validation stats every x batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint =  "checkpoint_A_10x10RND.pth.tar" # path to checkpoint, None if none

# load up data class:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Load train data set:
Data= TextDataset(txt_dir= os.getcwd() + train_dir, 
               vocab_dir= os.getcwd() + vocab_dir,
               height= 210, width= 210, max_lines= 10, font_size=12, ppl=7,
               forceRGB=True, V_spacing=11, train= True, input_method= "words")#, save_img= True, plot_grid= True)

word_map= Data.vocab_dict # dictionary of vocabulary and indices
Ntokens= len(word_map) # number of unique word tokens

# create separate set for validation:
ValidData= TextDataset(txt_dir= os.getcwd() + valid_dir,
               vocab_dir= os.getcwd() + vocab_dir,
               height= 210, width= 210, max_lines= 10, font_size=12, ppl=7,
               forceRGB=True, V_spacing=11, train= False, input_method= "words")# save_img= True, plot_grid= True)


def main():
    """
    Training and validation.
    """

    global checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, TrainModel
    
    
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
    
    print("Learning rate: %f\n" % (decoder_optimizer.param_groups[0]['lr']))
    
    # Epochs
    for epoch in range(start_epoch, epochs):
        
        start_epoch= time.time()
        adjust_learning_rate(decoder_optimizer, 0.25)
        adjust_learning_rate(encoder_optimizer, 0.25)
        
        if TrainModel:
            # One epoch's training
            train(train_loader= train_loader,
                  encoder= encoder,
                  decoder= decoder,
                  criterion= criterion,
                  encoder_optimizer= encoder_optimizer,
                  decoder_optimizer= decoder_optimizer,
                  epoch= epoch)
            
            # save model before validation (in case it crashes)
            save_checkpoint("A_"+ data_name, epoch, encoder, decoder, encoder_optimizer,
                            decoder_optimizer)
            
        # One epoch's validation
        validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

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
    #top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, string) in enumerate(train_loader):
        
#        state = {'imgs': imgs,
#             'caps': caps,
#             'caplens': caplens,
#             'string': string}
#        filename = 'input' + '.pth.tar'
#        torch.save(state, filename)
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps= caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs) # (batch_size, H_feature, W_feature, N_channels)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:] # (bacth_size, max_cap_len-1)

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, indx_scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, indx_targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        #loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

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
            
        # extract the element-wise values in the batch:
        list_targets= unflatten(targets, indx_targets, decode_lengths, False)
        list_scores= unflatten(scores, indx_scores, decode_lengths, True)
        
        # Keep track of metrics
        acc, mistakes, right, wrong_ind = accuracyTrain(list_scores, list_targets, word_map)
        losses.update(loss.item(), sum(decode_lengths))
        #top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time: {batch_time.avg:.3f}\t'
                  'Data Load Time: {data_time.avg:.3f}\t'
                  'Loss: {loss.avg:.4f}\t'
                  'Accuracy: {meanAcc:.3f} ({minAcc:.3f} - {maxAcc:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          meanAcc= np.mean(acc),
                                                                          minAcc= np.min(acc),
                                                                          maxAcc= np.max(acc)))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    """
    
    global AllAcc, AllMistakes, AllRight, AllWrong_ind, AllAttn_corr, AllCorrect, AllPredicted, AllPos
    
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    #top5accs = AverageMeter()
    AllAcc= [] # collect all accuracies to caclulate avg for validation stage
    AllMistakes= [] # collect all wrongly predicted tokens by the model (for testing)
    AllRight= [] # collect the true token that should have been predicted
    AllWrong_ind = [] # collecr token position for all wrongly predicted tokens by the model (for testing)
    AllAttn_corr = [] # attentional correctness for each token
    AllCorrect= []
    AllPredicted= []
    AllPos= []
    start = time.time()
    
    # Batches
    for i, (imgs, caps, caplens, rawImage, coords) in enumerate(val_loader): #, coords
        
#        state = {'imgs': imgs,
#             'caps': caps,
#             'caplens': caplens,
#             'rawImage': rawImage,
#             'coords': coords}
#        filename = 'VALinput' + '.pth.tar'
#        torch.save(state, filename)

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
        scores, indx_scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, indx_targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        alphas, indx_alphas = pack_padded_sequence(alphas, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        #loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        
        # extract the element-wise values in the batch:
        list_targets= unflatten(targets, indx_targets, decode_lengths, False)
        list_scores= unflatten(scores, indx_scores, decode_lengths, True)
        list_alphas= unflatten(alphas, indx_alphas, decode_lengths, True)

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        acc, mistakes, right, wrong_ind, attn_corr, correct_item, predicted_item, pos = accuracy(list_scores, list_targets, list_alphas, word_map, coords, sort_ind)
        #top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        AllAcc.append(acc)
        AllMistakes.extend(mistakes)
        AllRight.extend(right)
        AllWrong_ind.extend(wrong_ind)
        AllAttn_corr.extend(attn_corr)
        AllCorrect.extend(correct_item)
        AllPredicted.extend(predicted_item)
        AllPos.extend(pos)
        
        if save_worst_image:
            
            pos= min(enumerate(acc), key=itemgetter(1))[0] # find position of worst image in the batch:
            actual_pos= sort_ind[pos].item() # get actual position in original batch (due to sorting in LSTM):
            
            rawImage= rawImage.numpy() # re-code raw images as numpy array:
            image= rawImage[actual_pos,:,:] # take image from the bacth
            image = Image.fromarray(image) # conver to PIL image format
            image.save("pics/Batch"+ str(i)+ ".jpeg")
            
            # append wrong tokens only for the worst image in the batch:
            #AllWrong.extend(wrong[actual_pos]) # collect all wrongly predicted tokens by the model (for testing)
            #AllWrong_ind.extend(wrong_ind[actual_pos])
        
        if save_animation:
            generateGIF(rawImage, list_alphas, list_scores, list_targets, sort_ind,
                            word_map, filename= 'gifRND/'+ 'B'+ str(i)+ '.gif')#, upscale= 21, reshapeDim= 10)

        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time: {batch_time.avg:.3f}\t'
                  'Loss: {loss.avg:.4f}\t'
                  'Accuracy: {meanAcc:.3f} ({minAcc:.3f} - {maxAcc:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                            loss=losses, meanAcc= np.mean(acc),
                                                                            minAcc= np.mean(acc), maxAcc= np.max(acc)))
            print('Attentional correctness:\t M= {meanAttn:.3f} \t SD= {SDAttn:.3f} \t ({minAttn:.3f} - {maxAttn:.3f}) \n'.format(
                    meanAttn= np.mean(attn_corr), SDAttn= np.std(attn_corr), minAttn= np.min(attn_corr), maxAttn= np.max(attn_corr)))

    print(
        '\n * LOSS - {loss.avg:.3f}, ACCURACY - {meanAcc:.3f} ({minAcc:.3f} - {maxAcc:.3f}) \n'.format(
            loss=losses,
            meanAcc= np.mean(AllAcc), minAcc= np.min(AllAcc), maxAcc= np.max(AllAcc)))


if __name__ == '__main__':
    main()
    
    # calculate some additional measures:
    Token_length= [] # token length
    Error= []
    for i in range(len(AllCorrect)):
        Token_length.append(len(AllCorrect[i]))
        if AllCorrect[i]== AllPredicted[i]:
            Error.append(0)
        else:
            Error.append(1)
            
    df= pd.DataFrame({'pos': AllPos, 'correct': AllCorrect, 'predicted': AllPredicted, 'attn_corr': AllAttn_corr,
                      'error': Error, 'token_len': Token_length})
    df.to_excel(result_filename+".xls")
#    df= pd.DataFrame({'Wrong': AllMistakes, 'Right': AllRight})
#    df.to_excel("Val_results.xls")
#    
    AllWrong_ind = [item for sublist in AllWrong_ind for item in sublist] # turn into 1 list
    df2= pd.DataFrame({'Index': AllWrong_ind})
    df2.to_excel("Val_index_errors.xls")
    
    # Plot histogram:
    import matplotlib.pyplot as plt

    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x= AllWrong_ind, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Serial position of token')
    plt.ylabel('Frequency')
    plt.title('Frequency of errors for each word position in the text')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
       