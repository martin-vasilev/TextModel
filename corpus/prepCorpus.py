# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:20:13 2018

@author: mvasilev
"""

minChar= 6*60

import os
os.chdir('C:\\Users\\mvasilev\\TextModel\\corpus')


### Open output file for writing:
#file = open("COCA_academ.txt", "w") 
#file = open("COCA_fict.txt", "w") 
#file = open("COCA_mag.txt", "w") 
file = open("COCA_news.txt", "w") 

# Find all files in directory:
#allFiles= os. listdir("COCA\\acad")
#allFiles= os. listdir("COCA\\fict")
#allFiles= os. listdir("COCA\\mag")
allFiles= os. listdir("COCA\\news")

#subdir= "acad"
#subdir= "fict"
#subdir= "mag"
subdir= "news"

for k in range(len(allFiles)):
    
    print(allFiles[k])
    # load txt data:
    with open("COCA\\" + subdir +"\\" + allFiles[k], 'r') as myfile:
        data= myfile.read()
        
    ## clean things up:
    # remove remaining html tags:
    data = data.replace("<p>", "")
    data = data.replace("</p>", "")
    data = data.replace("<P>", "")
    data = data.replace("</P>", "")
    data = data.replace("<>", "")
    
    
    # get passages:
    text= data.split('\n')
    if text[0]== "":
        del text[0]
    
    for i in range(len(text)): # for each story..
        line= text[i]
        segment= line.split('@ @ @ @ @ @ @ @ @ @')
    
        for j in range(len(segment)): # for each segment..
            useString= segment[j]
            
            # locate piece of text to cut:
            start= useString.find(".") # first dot        
            end= useString.rfind(".")# last dot
            
            if start ==-1 or start == end:
                start= useString.find("?")
                if start== -1:
                    start= useString.find("!")
                    
            if end ==-1 or start == end:
                end= useString.find("?")
                if end== -1:
                    end= useString.find("!")
            
            # take string we need:
            string= useString[start+1:end+1]
                
            if len(string) >= minChar:
                if string[0]== " ":
                    string= string[1:]
                if string[0].isupper():  # save it only if it starts in upper case (i.e., new sentence)
                    import re
                    num= map(int, re.findall(r'\d', string))
                    if len(num) < len(string)/20: # if numbers in text are max 5% of all chars
                        if string[1]!= ".": # some reference entries
                            if not "Journal" in string and not "ISBN" in string and not "doi:" in string and not "Press" in string and not "conference" in string:
                                
                                string = string.replace("  ", " ")
                                string = string.replace(" ,", ",")
                                string = string.replace(" .", ".")
                                string = string.replace(" '", "'")
                                string = string.replace(" :", ":")
                                string = string.replace(" ?", "?")
                                string = string.replace(" !", "!")
                                string = string.replace(" ;", ";")
                                string = string.replace(" -", "-")
                                string = string.replace("( ", "(")
                                string = string.replace(" )", ")")
                                #string = string.replace('" ', '"')
                                #string = string.replace(' "', '" ')
                                
                                if len(string) >= minChar:
                                    file.write(string)
                                    file.write("\n")
           # else:
                #print("Not enough chars.\n")
file.close()