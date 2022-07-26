#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:52:14 2021

@author: chay
"""
import pandas as pd
import re
import numpy as np

import nltk
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
n=WordNetLemmatizer()

'''
year_list = ['1981','1982','1983','1984','1985','1986','1987','1988','1989',
             '1990','1991','1992','1993','1994','1995','1996','1997','1998','1999',
             '2000','2001','2002','2003','2004','2005','2006','2007','2008','2009',
             '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
             '2020', '2021']
'''

year_list = ['1981_1','1981_2',
                  '1982_1','1982_2',
                  '1983_1','1983_2',
                  '1984_1','1984_2',
                  '1985_1','1985_2',
                  '1986_1','1986_2',
                  '1987_1','1987_2',
                  '1988_1','1988_2',
                  '1989_1','1989_2',
                  '1990_1','1990_2',
                  '1991_1','1991_2',
                  '1992_1','1992_2',
                  '1993_1','1993_2',
                  '1994_1','1994_2',
                  '1995_1','1995_2',
                  '1996_1','1996_2',
                  '1997_1','1997_2',
                  '1998_1','1998_2',
                  '1999_1','1999_2',
                  '2000_1','2000_2',
                  '2001_1','2001_2',
                  '2002_1','2002_2',
                  '2003_1','2003_2',
                  '2004_1','2004_2',
                  '2005_1','2005_2',
                  '2006_1','2006_2',
                  '2007_1','2007_2',
                  '2008_1','2008_2',
                  '2009_1','2009_2',
                  '2010_1','2010_2',
                  '2011_1','2011_2',
                  '2012_1','2012_2',
                  '2013_1','2013_2',
                  '2014_1','2014_2',
                  '2015_1','2015_2',
                  '2016_1','2016_2',
                  '2017_1','2017_2',
                  '2018_1','2018_2',
                  '2019_1','2019_2',
                  '2020_1','2020_2',
                  '2021_1','2021_2']


def pre_process(text):
    
    text = text.lower()
    text = n.lemmatize(text)
    return text
    
keywords=[]
df = pd.read_excel('word2vec.xlsx', sheet_name= 'cluster_results', index_col=0)
words = df['word']
    
for word in words:
    
    word = word.lower()
    keywords.append(word)
        
result_year=[]
abstract=[]


for year in year_list:
    
    file_name = year+'.xlsx'
    df_1 = pd.read_excel(file_name,index_col=0, engine ='openpyxl')
    rows=df_1.shape[0]
    print(year)

    result_each_paper=[]
    result_all_paper=[]
    for i in range(rows):
        
        if df_1['sjr'][i] == 'n':
            #print(" no weight")
            w = 0 
        else: 
            w = df_1['weighted matrix_3:7'][i]
        
        abstract = df_1['description'][i]

        if abstract is np.nan:
            print(" no abstract")
           
        else:
            clean_abstract = pre_process(abstract)
            result=[]
            
            for keyword in keywords:
                re.findall(keyword,clean_abstract)
            
                result.append(len(re.findall(keyword,clean_abstract))*w)
        result_each_paper.append(result)
    result_all_paper = [sum(x) for x in zip(*result_each_paper)]
    result_year.append(result_all_paper)

#print(result_year)

outputFileName = 'keyword_frequency_half_year_weight.xlsx' 
df_2= pd.DataFrame(result_year).transpose()  #df
df_2.to_excel(outputFileName,sheet_name='counter')  
 

     
     
     
     
     