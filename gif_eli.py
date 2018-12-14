# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 00:19:07 2018

@author: marti
"""

def generateGIF(rawImage, alphas, list_scores, list_targets, word_map, filename= 'gif/test.gif'):
    # code adapted from: https://eli.thegreenplace.net/2016/drawing-animated-gifs-with-matplotlib/    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import skimage.transform
    
    # tokens from model:
    _, token= list_scores[0].max(dim= 1)
    words= list(word_map.keys())
    
    # correct word tokens in image:
    correct= list_targets[0].long()
    
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    
    
    ax.imshow(rawImage)
    #ax.axis('off')
    ax.tick_params(axis='both', left='off', top='off', right='off', bottom='on',
                   labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    
    def update(i):
        label = '({0})             {word}'.format(i, word= words[token[i]])
        print(label)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        ax.set_xlabel(label)
        ax.tick_params(axis='x', colors='white') # psst.. this is a workaround
        
        ax.imshow(rawImage)
        a = alphas[0, i, :].detach().cpu().numpy().reshape(10, 10)
        alpha = skimage.transform.pyramid_expand(a, upscale=21, sigma=8)
        ax.imshow(alpha, alpha=0.5)
        
        # display correct tokens in green and wrong ones in red:
        if words[token[i]]== words[correct[i]]:
            ax.xaxis.label.set_color('green')
        else:
            ax.xaxis.label.set_color('red')
            
        # add grid lines:
        coords = [21, 21*2, 21*3, 21*4, 21*5, 21*6, 21*7, 21*8, 21*9, 21*10]
        for xc in coords:
            ax.axvline(x=xc)
            ax.axhline(y=xc)
    
        return ax
    
    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames= np.arange(0, alphas.size(1)), interval=1000, repeat= True, repeat_delay= 1000)
    anim.save(filename, dpi=80, writer='imagemagick')