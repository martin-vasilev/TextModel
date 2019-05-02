# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:43:20 2018

@author: Martin R. Vasilev

Adapted code from: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
"""

import numpy as np
import torch
from itertools import chain
import skimage.transform


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


def save_checkpoint(data_name, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    """
    state = {'epoch': epoch,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
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


def accuracy(list_scores, list_targets, list_alphas, word_map, coords, sort_ind):
    """
    Calculates accuracy for each image in the batch (including attention correctness)

    Input:
        list_scores: a list of token predictions for each image
        list_targets: a list of actual targets in each image (ground truth)
    """
    
    acc= [] # list to hold individual image accuracies
    wrong_ind= []
    mistakes = []
    right= []
    attn_corr= []
    word_len= [] # length of the word in pixels
    correct_item= []# the ground truth for each token
    predicted_item= []  # the token predicted by the model
    pos= [] # position of token in image
    
    for i in range(len(list_scores)): # for each image in batch..
        correct= list_targets[i].long() # which are the actual correct tokens for image?
        _, token= list_scores[i].max(dim= 1) # which are the predicted tokens by the model?
        
        # calculate accuracy for image (predicted/correct)*100
        acc.append(((torch.eq(correct, token).sum().item())/len(correct))*100)
        
        words= list(word_map.keys()) # word strings
        comp= torch.eq(correct, token)
        mistakes_ind= (comp == 0).nonzero()
        mistakes_token= token[mistakes_ind]
        mistakes_right= correct[mistakes_ind]
        
        for j in range(len(mistakes_token)): # which are the mistaken words?
            mistakes.append(words[mistakes_token[j]])
        
        for j in range(len(mistakes_right)): # which were the actual correct words?
            right.append(words[mistakes_right[j]])
        
        wrong_ind.append(list(chain(*mistakes_ind.tolist())))
        #wrong.append(list(chain(*mistakes_token.tolist())))
        
        # Attentional correctness:
        alpha= list_alphas[i] # alpha activations for current image
        ntokens= len(correct)-1 # -1 because of <end>
        alpha= alpha[:ntokens,:]
        
        for k in range(ntokens):
            # alphas for current token (resize to input image size):
            alpha_k = alpha[k, :].detach().cpu().numpy().reshape(10, 10) # (10, 10)
            alpha_k = skimage.transform.pyramid_expand(alpha_k, upscale= 21) # (210, 210)
            dims= coords[sort_ind[i].item(), k,:, :] # xy dimensions for token on image (1,4)
            dims= dims[0, :] # (4)
            
            # normalize image so that it sums up to 1:
            alpha_k= alpha_k/np.sum(alpha_k)
            
            # take alpha activation only within the token boundaries:
            x1= dims[0].item() # x start
            x2= dims[2].item() # x end
            y1= dims[1].item() # y start
            y2= dims[3].item() # y end
            alpha_token= alpha_k[y1:y2, x1:x2] # subset image
        
            attn_corr.append(np.sum(alpha_token)) # save attentional correctness for later analysis
            correct_item.append(words[correct[k].item()])
            predicted_item.append(words[token[k].item()])
            pos.append(k)
        
    return acc, mistakes, right, wrong_ind, attn_corr, correct_item, predicted_item, pos



def accuracyTrain(list_scores, list_targets, word_map):
    """
    Calculates accuracy for each image in the batch
    Light-wight version for training (to reduce training times)

    Input:
        list_scores: a list of token predictions for each image
        list_targets: a list of actual targets in each image (ground truth)
    """
    
    acc= [] # list to hold individual image accuracies
    wrong_ind= []
    mistakes = []
    right= []
    
    for i in range(len(list_scores)):
        correct= list_targets[i].long() # which are the actual correct tokens for image?
        _, token= list_scores[i].max(dim= 1) # which are the predicted tokens by the model?
        
        # calculate accuracy for image (predicted/correct)*100
        acc.append(((torch.eq(correct, token).sum().item())/len(correct))*100)
        
        words= list(word_map.keys()) # word strings
        comp= torch.eq(correct, token)
        mistakes_ind= (comp == 0).nonzero()
        mistakes_token= token[mistakes_ind]
        mistakes_right= correct[mistakes_ind]
        
        for j in range(len(mistakes_token)): # which are the mistaken words?
            mistakes.append(words[mistakes_token[j]])
        
        for j in range(len(mistakes_right)): # which were the actual correct words?
            right.append(words[mistakes_right[j]])
        
        wrong_ind.append(list(chain(*mistakes_ind.tolist())))
        #wrong.append(list(chain(*mistakes_token.tolist())))
        
    return acc, mistakes, right, wrong_ind



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

# generates an animated Gif of the model's performance:
def generateGIF(rawImage, list_alphas, list_scores, list_targets, sort_ind, word_map, filename= 'gif/test.gif', upscale= 21,
                reshapeDim= 10, plot_grid= False):
    # code adapted from: https://eli.thegreenplace.net/2016/drawing-animated-gifs-with-matplotlib/  
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import skimage.transform
    
    rawImage= rawImage[sort_ind[0].item(),:,:]
    alphas= list_alphas[0]
    
    # tokens from model:
    _, token= list_scores[0].max(dim= 1)
    words= list(word_map.keys())
    
    # correct word tokens in image:
    correct= list_targets[0].long()
    
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    #fig.set_tight_layout(False)
    
    ax.imshow(rawImage)
    #ax.axis('off')
    ax.tick_params(axis='both', left='off', top='off', right='off', bottom='on',
                   labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    
    def update(i):
        label = '({0})             {word}'.format(i, word= words[token[i]])
        print(i)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        ax.set_xlabel(label)
        ax.tick_params(axis='x', colors='white') # psst.. this is a workaround
        
        ax.imshow(rawImage, cmap='Greys_r') # cmap to get read of wird grayscale artefact (yellow background)
        a = alphas[i, :].detach().cpu().numpy().reshape(reshapeDim, reshapeDim)
        alpha = skimage.transform.pyramid_expand(a, upscale= upscale)
        ax.imshow(alpha, alpha=0.6)
        
        # display correct tokens in green and wrong ones in red:
        if words[token[i]]== words[correct[i]]:
            ax.xaxis.label.set_color('green')
        else:
            ax.xaxis.label.set_color('red')
        
        if plot_grid:
            # add grid lines:
            coords = list(range(0, reshapeDim*upscale, upscale))
            for xc in coords:
                ax.axvline(x=xc)
                ax.axhline(y=xc)
    
        return ax
    
    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames= np.arange(0, alphas.size(0)), interval=750, repeat= True, repeat_delay= 1000)
    anim.save(filename, dpi=80, writer='imagemagick')