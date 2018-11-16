# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:19:29 2018

@author: marti
"""

import wikipedia
import os
import sys
os.chdir('D:\\R\\TextModel')
sys.path.insert(0, './corpus')


file = open("corpus/wiki_corpus2.txt", "a", encoding='utf-8')

with open("corpus/all_articles.txt", 'r', encoding='utf-8') as myfile:
    data= myfile.read()
    titles= data.split('\n')
    myfile.close()
    
file2 = open("corpus/all_articles.txt", "a", encoding='utf-8')


for i in range(500):
    try:
        pg= wikipedia.page(wikipedia.random(pages=1)) # get a random article
        if not pg.title in titles: # check if we already have it:
            titles.append(pg.title)
            file2.write(pg.title) # save so we can keep track of what we have
            file2.write("\n")
            
            text= pg.content
            file.write("\n")
            file.write("@@@@@@")
            file.write("\n")
            file.write(pg.title)
            file.write("\n")
            file.write(text) # save article
            file.write("\n")
        else:
            print("already available")
    except:
        pass
    print(i)
    
file.close()
file2.close()      