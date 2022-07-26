#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:30:29 2019

@author: chay
"""
from pybliometrics.scopus import ScopusSearch
import pandas as pd
import requests
import pybliometrics
from pybliometrics.scopus.utils import config


config['Authentication']['InstToken'] = ''
print(config['Authentication']['InstToken'])

query = 'TITLE-ABS-KEY(antenna)'
#query = 'AF-ID("Queen Mary University of London" 60022109) OR AF-ID("Barts and The London School of Medicine and Dentistry" 60021435) OR AF-ID("Faculty of Humanities and Social Sciences" 60162684) OR AF-ID("School of Business and Management" 60160411) OR AF-ID("MRC - Asthma UK Centre in Allergic Mechanisms of Asthma" 60002517) OR AF-ID("NIHR Research Design Service London" 60177639) OR AF-ID("School of Biological and Chemical Sciences Queen Mary University of London" 60105356) OR AF-ID("The CRUK City of London Centre" 60176026) OR AF-ID("UK Grid for Particle Physics" 60017096) AND ( LIMIT-TO ( PUBYEAR,2021) OR LIMIT-TO ( PUBYEAR,2020) OR LIMIT-TO ( PUBYEAR,2019) OR LIMIT-TO ( PUBYEAR,2018) OR LIMIT-TO ( PUBYEAR,2017) )'


s = ScopusSearch(query, 
                 download=True, # save the results 
                 verbose=True)  # present the process

df_s = pd.DataFrame(s.results)
df_s.head()
df_s.to_csv("antenna_all.csv")    
