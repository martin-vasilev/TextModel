# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:30:42 2018

@author: mvasilev
"""


class Corpus(object):
    
    @staticmethod
    # take N most frequent tokens from SUBTLEX-UK database:
    def SUBTLEX(N, dir= "C:\Users\mvasilev\TextModel\corpus\SUBTLEX-UK.txt"):
        with open(dir, 'r') as f:
            corpustxt= f.readlines()
        
        import re
        
        token= []
        freq= []
        with open(dir, 'r') as f:
            corpustxt= f.readlines()
            
            for i in range(len(corpustxt)-2):
                if i>0:
                    string= re.split(r'\t+', corpustxt[i])
                    token.append(string[0])
        
                    freq.append(string[5])
        
        # sort tokens by frequency:
        Z = zip(freq, token)
        Z.sort(reverse= True)
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
        newstr= newstr.replace('-', '')
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
        
        
        newstr = ''.join([i for i in newstr if not i.isdigit()])# remove numbers
        
        if lower:
            newstr= newstr.lower()# make all letters lowercase
        
        # separate into words:
        wrds= newstr.split(" ")
        wrds = filter(None, wrds)
        
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
                    