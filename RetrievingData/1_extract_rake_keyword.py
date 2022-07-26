#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:29:56 2021

@author: chay
"""

import csv
import re
import nltk
import pandas as pd


from nltk.corpus import stopwords  
from rake_nltk import Metric, Rake
stop_words = set(stopwords.words('english'))


df = pd.read_excel('antenna_data.xlsx', sheet_name='antenna_all', header =0) 
abstracts = df['description']

r = Rake(language="ENGLISH", stopwords=stop_words, ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO, min_length=2, max_length=2)
                #punctuations=<string of puntuations to ignore>,
results=[]
number_abstract = abstracts.size



j=0
for j in range(number_abstract+1):  

    try:
        r.extract_keywords_from_text(abstracts[j])
        keyword = r.get_ranked_phrases()
    except:
        keyword = None
        

    results.append(keyword)

result= { 
            
                "Rake Keyword" : results
         }

    
df_1 = pd.DataFrame(result)  #df
   
outputFileName = 'rake_keywords.xlsx'  
df_1.to_excel(outputFileName,sheet_name='sheet1')    




