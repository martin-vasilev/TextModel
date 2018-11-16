# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:30:42 2018

@author: mvasilev
"""    

class Corpus(object):
    
    @staticmethod
    # take N most frequent tokens from SUBTLEX-UK database:
    def SUBTLEX(N, dir= "corpus\SUBTLEX-US.txt", UK=False):        
        import re
        import sys
        
        token= []
        freq= []
        with open(dir, 'r') as f:
            corpustxt= f.readlines()
            
            for i in range(len(corpustxt)-2):
                if i>0:
                    string= re.split(r'\t+', corpustxt[i])
                    token.append(string[0])
                    
                    if not UK:
                        freq.append(string[14])
                    else:
                        freq.append(string[5])
        
        # sort tokens by frequency:
        if sys.version_info[0] < 3:
            Z = zip(freq, token)
            Z.sort(reverse= True)
        else:
            Z= sorted(zip(freq, token), reverse= True)
        token_sorted = [x for y, x in Z]
        freq.sort(reverse= True)
        out= token_sorted[0:N]
        print("Min freqency in selection: %s (Zipf)" % (freq[N]))
        return out
    
    @staticmethod
    # Get word tokens from a string:
    def strip(string, lower= True):
        wrds= string.split(" ")
        newstr= string
        
        # remove characters from string:
        newstr= newstr.replace(",", "")
        newstr= newstr.replace(".", "")
        newstr= newstr.replace('"', '')
        newstr= newstr.replace('#', '')
        newstr= newstr.replace(':', '')
        newstr= newstr.replace(';', '')
        #newstr= newstr.replace('-', '')
        newstr= newstr.replace('!', '')
        newstr= newstr.replace('?', '')
        newstr= newstr.replace('*', '')
        newstr= newstr.replace('&', '')
        newstr= newstr.replace('^', '')
        newstr= newstr.replace('%', '')
        newstr= newstr.replace('(', '')
        newstr= newstr.replace(')', '')
        newstr= newstr.replace('@', '')
        newstr= newstr.replace('~', '')
        newstr= newstr.replace('<', '')
        newstr= newstr.replace('>', '')
        newstr= newstr.replace('+', '')
        newstr= newstr.replace('}', '')
        newstr= newstr.replace('{', '')
        newstr= newstr.replace('[', '')
        newstr= newstr.replace(']', '')
        newstr= newstr.replace('$', '')
        newstr= newstr.replace('/', '')
        newstr= newstr.replace('|', '')
        newstr= newstr.replace("' ", " ")
        
        ### language abreviations and so on:
        newstr = newstr.replace("wo n't", "will not")
        newstr= newstr.replace("'s", " s")
        newstr= newstr.replace("s'", " s")
        newstr= newstr.replace("n't", "not")
        newstr= newstr.replace("'re", " are")
        newstr= newstr.replace("'ve", " have")
        newstr= newstr.replace("'ll", " will")
        newstr= newstr.replace("'d", " would")
        newstr= newstr.replace("i'm", "I am")
        newstr= newstr.replace("'m", " am")
        
         
        newstr = ''.join([i for i in newstr if not i.isdigit()])# remove numbers
        if "-" in newstr: # for compound words
            newstr= newstr.split("-") 
            newstr= ' '.join(newstr)
        if "/" in newstr:
            newstr= newstr.split("/")
            newstr= ' '.join(newstr)
        
        if lower:
            newstr= newstr.lower()# make all letters lowercase
        
        # separate into words:
        wrds= newstr.split(" ")
        wrds = list(filter(None, wrds))
        
        if "i" in wrds and lower:
            indices = [i for i, x in enumerate(wrds) if x == "i"]
            for k in range(len(indices)):
                wrds[indices[k]]= "i".upper() # I is upper case in subtlex-uk
            
        return(wrds)
    
    @staticmethod
    def find_substr(substring, string):
        Done= False
        indices= []
        while not Done:
            if substring in string:
                att= string.index(substring)
                indices.append(att)
                string= string[0:att] + "@"*len(substring) + string[att+ len(substring):]
            else:
                Done= True

        return indices
    
    @staticmethod
    def unique_list(list):
        uniq_list = []
        uniq_set = set()
        for item in list:
           if item not in uniq_set:
                uniq_list.append(item)
                uniq_set.add(item)
        return uniq_list