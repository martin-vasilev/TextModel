# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:30:42 2018

@author: Martin R. Vasilev
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
        #newstr= newstr.replace('/', ' ')
        newstr= newstr.replace('|', '')
        newstr= newstr.replace("' ", " ")
        
        ### language abreviations and so on:
        newstr = newstr.replace("wo n't", "will not")
        newstr= newstr.replace("'s", " s")
        newstr= newstr.replace("s'", " s")
        newstr= newstr.replace("n't", "not")
        newstr= newstr.replace("canot", "cannot")
        newstr= newstr.replace("arenot", "are not")
        newstr= newstr.replace("didnot", "did not")
        newstr= newstr.replace("donot", "do not")
        newstr= newstr.replace("wouldnot", "would not")
        newstr= newstr.replace("wasnot", "was not")
        newstr= newstr.replace("isnot", "is not")
        newstr= newstr.replace("doesnot", "does not")
        newstr= newstr.replace("hadnot", "had not")
        newstr= newstr.replace("werenot", "were not")
        newstr= newstr.replace("wonot", "will not")
        newstr= newstr.replace("couldnot", "could not")
        newstr= newstr.replace("shouldnot", "should not")
        newstr= newstr.replace("ainot", "are not")
        newstr= newstr.replace("neednot", "need not")
        newstr= newstr.replace("havenot", "have not")
        newstr= newstr.replace("hasnot", "has not")
        newstr= newstr.replace("mustnot", "has not")
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
        for k in range(len(wrds)): # to remove apostrophe at the end
            if wrds[k][len(wrds[k])-1]== "'":
                wrds[k]= wrds[k][0:len(wrds[k])-2]
        
        if "i" in wrds and lower:
            indices = [i for i, x in enumerate(wrds) if x == "i"]
            for k in range(len(indices)):
                wrds[indices[k]]= "i".upper() # I is upper case in subtlex-uk
            
        return(wrds)
    
    
    @staticmethod
    # Same as above, but included numbers and punctuation too
    def strip2(string, lower= True):
        
        def fix_num(string):
            num_set= ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            strs= ""
            for i in range(len(string)):
                if string[i] in num_set:
                    strs= strs + " "+ string[i]
                    if i+1<= len(string)-1:
                        if string[i+1]!= " ":
                            strs= strs+ " "
                else:
                    strs= strs+ string[i]
            
            return strs
        
        
        #wrds= string.split(" ")
        newstr= string
        
        # remove characters from string:
        newstr= newstr.replace(",", " , ")
        newstr= newstr.replace(".", " . ")
        newstr= newstr.replace('"', ' " ')
        newstr= newstr.replace('#', ' # ')
        newstr= newstr.replace(':', ' : ')
        newstr= newstr.replace(';', ' ; ')
        #newstr= newstr.replace('-', '')
        newstr= newstr.replace('!', ' ! ')
        newstr= newstr.replace('?', ' ? ')
        newstr= newstr.replace('*', ' * ')
        newstr= newstr.replace('&', ' & ')
        newstr= newstr.replace('^', ' ^ ')
        newstr= newstr.replace('%', ' % ')
        newstr= newstr.replace('(', ' ( ')
        newstr= newstr.replace(')', ' ) ')
        newstr= newstr.replace('@', ' @ ')
        newstr= newstr.replace('~', ' ~ ')
        newstr= newstr.replace('<', ' < ')
        newstr= newstr.replace('>', ' > ')
        newstr= newstr.replace('+', ' + ')
        newstr= newstr.replace('}', ' } ')
        newstr= newstr.replace('{', ' { ')
        newstr= newstr.replace('[', ' [ ')
        newstr= newstr.replace(']', ' ] ')
        newstr= newstr.replace('$', ' $ ')
        #newstr= newstr.replace('/', ' ')
        newstr= newstr.replace('|', ' | ')
        newstr= newstr.replace("' ", " ")
        
        ### language abreviations and so on:
        newstr = newstr.replace("wo n't", "will not")
        newstr= newstr.replace("'s", " s")
        newstr= newstr.replace("s'", " s")
        newstr= newstr.replace("n't", "not")
        newstr= newstr.replace("canot", "can not")
        newstr= newstr.replace("arenot", "are not")
        newstr= newstr.replace("didnot", "did not")
        newstr= newstr.replace("donot", "do not")
        newstr= newstr.replace("wouldnot", "would not")
        newstr= newstr.replace("wasnot", "was not")
        newstr= newstr.replace("isnot", "is not")
        newstr= newstr.replace("doesnot", "does not")
        newstr= newstr.replace("hadnot", "had not")
        newstr= newstr.replace("werenot", "were not")
        newstr= newstr.replace("wonot", "will not")
        newstr= newstr.replace("couldnot", "could not")
        newstr= newstr.replace("shouldnot", "should not")
        newstr= newstr.replace("ainot", "are not")
        newstr= newstr.replace("neednot", "need not")
        newstr= newstr.replace("havenot", "have not")
        newstr= newstr.replace("hasnot", "has not")
        newstr= newstr.replace("mustnot", "has not")
        newstr= newstr.replace("'re", " are")
        newstr= newstr.replace("'ve", " have")
        newstr= newstr.replace("'ll", " will")
        newstr= newstr.replace("'d", " would")
        newstr= newstr.replace("i'm", "I am")
        newstr= newstr.replace("'m", " am")
        newstr= newstr.replace("cannot", "can not")
        
        newstr= fix_num(newstr)
        #newstr = ''.join([i for i in newstr if not i.isdigit()])# remove numbers
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
        for k in range(len(wrds)): # to remove apostrophe at the end
            if wrds[k][len(wrds[k])-1]== "'":
                wrds[k]= wrds[k][0:len(wrds[k])-2]
        
        if "i" in wrds and lower:
            indices = [i for i, x in enumerate(wrds) if x == "i"]
            for k in range(len(indices)):
                wrds[indices[k]]= "i".upper() # I is upper case in subtlex-uk
            
        return(wrds)        