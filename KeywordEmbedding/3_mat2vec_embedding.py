#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:29:12 2021

@author: chay
"""

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import logging
from scipy.spatial.distance import cosine
import pandas as pd

from scipy.spatial.distance import cosine

model = Word2Vec.load("models/pretrained_embeddings")
first_vector_text=[]
second_vector_text=[]
multi_word_vector=[]
first_word_text=[]
second_word_text=[]
multi_word_text=[]
word_list=[]
vector_list=[]


df = pd.read_excel('keyword_dic_antenna.xlsx', sheet_name='dict_final', index_col=0)
word_pool = df['dup']

counter=0

def find_vector_value(word):

    if word in word_list:
        index = word_list.index(word)
        vector = vector_list[index]

    else:
        #print("Not found")
        vector=0
        
    return vector

for word in word_pool:
  
    try: 
        word_vectors = np.array([model[word]])
        #word_vectors = [model[word]]
        word_list.append(word)
        vector_list.append(word_vectors)
    
    except:
        print(word)
        #counter = counter+1   
        
df = pd.read_excel('keyword_dic_antenna.xlsx', sheet_name='keyword_dic_final', index_col=0)
multi_word_pool = df['word']

for multi_word in multi_word_pool:
    try:
        split = multi_word.split(' ')
        first_word = split[0]
        second_word = split[1]
        first_vector = find_vector_value(first_word)
        second_vector = find_vector_value(second_word)
        final_vector = first_vector + second_vector
        
        first_word_text.append(first_word)
        second_word_text.append(second_word)
        multi_word_text.append(multi_word)
        first_vector_text.append(first_vector)
        second_vector_text.append(second_vector)
        multi_word_vector.append(final_vector)
        
        
    except:
        final_vector = find_vector_value(multi_word)
        
        second_word=None
        second_vector=None

        first_word_text.append(multi_word)
        second_word_text.append(second_word)
        multi_word_text.append(multi_word)
        first_vector_text.append(first_vector)
        second_vector_text.append(second_vector)
        multi_word_vector.append(final_vector)
        counter = counter+1  
        
print(counter)



result= { 
                "first_word" : first_word_text,
                "second_word" : second_word_text,
                "multi_word" : multi_word_text,
                "first_vector" : first_vector_text,
                "second_vector" : second_vector_text,
                "multi_word_vector" : multi_word_vector
            }

df = pd.DataFrame(result)  #df
outputFileName = 'word2vec.xlsx'  
df.to_excel(outputFileName,sheet_name='sheet1') 








