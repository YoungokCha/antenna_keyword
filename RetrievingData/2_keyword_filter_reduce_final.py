#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:09:39 2019

@author: chay
"""
import pandas as pd
import csv
import re
import nltk
from nltk.corpus import stopwords  
nltk.download('wordnet') 
from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words('english'))


file_name = 'antenna_data.xlsx'
df = pd.read_excel(file_name, index_col=0 )

title = df['title']
author = df['authkeywords']
rake = df['Rake Keyword']

lemma_sum_word=[]
keyword_candidate=[]
total_paper=0

for i in range(len(title)):

    author_keyword =[]
    rake_keyword =[]
    title_split=[]
    temp_author=''
    rake_split=''
    temp_rake=''
    author_split=''

    title_split = title[i].split()
    title_split = [item.lower() for item in title_split]

    try:
        author_split = author[i].split("| ")
        for j in range(len(author_split)):
            temp_author=author_split[j].strip()
            temp_author = temp_author.lower()
            author_keyword.append(temp_author)
    except:
        None
        
    try:
        rake_split = rake[i].split(',')
        for k in range(len(rake_split)):
            temp_rake=rake_split[k].strip("[], , ' ' ")
            temp_rake = re.sub("(\\W)+"," ",temp_rake)
            rake_keyword.append(temp_rake)
    except:
        None
                
    if (author_keyword) :
        for word in rake_keyword:
            if(len(word.split())<3):
                if (word in author_keyword):
                    keyword_candidate.append(word)
    else:
        for word in rake_keyword:
            if(len(word.split())<3):
                if(len(set(word).intersection(set(title_split)))>1):
                    keyword_candidate.append(word)
                  
    total_paper = total_paper +1
        
print(total_paper)

n=WordNetLemmatizer()
for w in keyword_candidate:
    lemma_word=[]
    if w is not None:
        nltk_tokens = nltk.word_tokenize(w)
        for tk in nltk_tokens:
            lemma_word.append(n.lemmatize(tk))
    else:
        continue
    lemma_sum= ' '.join(lemma_word)
    lemma_sum_word.append(lemma_sum)
        

word_dic={}
word_sum = 0
for word in lemma_sum_word:

    if word in word_dic:
        word_dic[word] += 1
       
        
    else:
        word_dic[word] = 1


## sorted 
word_list = sorted(word_dic.items(), key=lambda x: x[1], reverse=True) #0: alphabetiacal , 1:frequent 
word_sum = sum(word_dic.values())
#print(word_list)   
print(len(word_list)) 
print(word_sum) 


with open('keyword_dic.csv', 'w') as f:
    writer = csv.writer(f)
    for val in word_list:
        writer.writerow([val])

  
        
        
        