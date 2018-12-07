# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:43:20 2018

@author: Martin R. Vasilev

Adapted code from: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
"""

import numpy as np
import torch


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, last_loss):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer,
             'last_loss': last_loss}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(list_scores, list_targets):
    """
    Calculates accuracy for each image in the batch

    Input:
        list_scores: a list of token predictions for each image
        list_targets: a list of actual targets in each image (ground truth)
    """
    
    acc= [] # list to hold individual image accuracies
    for i in range(len(list_scores)):
        correct= list_targets[i].long() # which are the actual correct tokens for image?
        _, token= list_scores[i].max(dim= 1) # which are the predicted token by the model?
        
        # calculate accuracy for image (predicted/correct)
        acc.append(((torch.eq(correct, token).sum().item())/len(correct))*100)
    return acc


def unflatten(tens, indx, lens, multidim= False):
    """
    Takes a flattened pad_packed_sequence and returns the item-wise numbers for each element in the batch
    
    Input:
        tens: flattened pytorch tensor after applying the pack_padded_sequence fun
        indx: tensor containing the number of batch elements for each sequence length
        lens: tensor containing the length of each batch element, sorted in descending order
        
    Output: a list of tensors containing each element of the batch
    """    
    
    import torch

    items= []
    curr= 0 
    
    if not multidim:
        
        for i in range(max(indx)):
            items.append(torch.zeros(lens[i]))
            
        for j in range(len(indx)):
            
            for k in range(indx[j]):
                items[k][j]= tens[curr]
                curr = curr+ 1
                #print(curr)
    else:
        
        for i in range(max(indx)):
            items.append(torch.zeros(lens[i], tens.size(1)))
    
        for j in range(len(indx)):
            
            for k in range(indx[j]):
                items[k][j, :]= tens[curr, :]
                curr = curr+ 1
                #print(curr)
    return items