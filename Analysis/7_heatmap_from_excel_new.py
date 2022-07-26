
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:22:05 2019

@author: chay
"""

#import pandas lib as pd 
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

# read 2nd sheet of an excel file 
df = pandas.read_excel('keyword_frequency_weight_year.xlsx', sheet_name = 'heatmap_em', index_col= 0)
fig = plt.figure(figsize=(20,25))
sns.set(font_scale=3)

#sns.heatmap(df, cmap="YlGnBu")
#sns.heatmap(df, cmap="Purples")
#sns.heatmap(df, cmap="Blues")
#sns.heatmap(df,  cmap="BuPu")
#sns.heatmap(df,  cmap="RdPu")
#sns.heatmap(df,  cmap="Reds")
#sns.heatmap(df,  cmap="PuBuGn")
sns.heatmap(df,  cmap="YlOrRd")
#plt.savefig('heatmap.png')
plt.show()


