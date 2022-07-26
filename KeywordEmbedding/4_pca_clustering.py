#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:19:48 2021

@author: chay
"""
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import random


import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
rcParams["font.family"] = "sans-serif"
rcParams['axes.linewidth'] = 2
rcParams['xtick.major.size'] = 6
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.size'] = 4
rcParams['xtick.labelsize'] = 14
rcParams['ytick.major.size'] = 6
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.size'] = 4
rcParams['ytick.labelsize'] = 14

df = pd.read_excel('word2vec.xlsx', sheet_name='word_vec_2415', index_col=0)
multi_word_vec = df['multi_word_vector']
words=df['multi_word']

split=[]
print(len(multi_word_vec))

for i in range(len(multi_word_vec)):
    split.append(multi_word_vec[i].split())
    

word_vec=np.array(split)
twodim = PCA().fit_transform(word_vec)[:,:2]
'''
# get k value (elbow)
iter=[]
for i in range(1,30):
    model = KMeans(n_clusters = i)
    model.fit(twodim)
    iter.append(model.inertia_)

s = pd.DataFrame(iter)

#plt.figure(figsize =(5,5))
sns.set_style("white")

#s.plot(kind='line', legend = False)
plt.plot(s)
plt.xlabel('Number of clusters, k')
plt.ylabel('Sum of squared distances')
plt.show()
'''

model = KMeans(n_clusters =7, random_state=42)
model.fit(twodim)
y_kmeans = model.fit_predict(twodim)
labels = model.labels_

plt.figure(figsize =(10,10))
sns.set_style("ticks")

#colormap = np.array(['red', 'orange','aqua', 'purple', 'green', 'magenta', 'blue'])
colormap = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
plt.scatter(twodim[:,0], twodim[:,1], c=colormap[model.labels_], s=40)

plt.xlabel('PC 1', size =15)
plt.ylabel('PC 2', size =15)



























