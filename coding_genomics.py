#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# In this Script we are coding genomics based in snps_and_effect_allels file

"""
Created on Sat Feb  9 19:08:59 2019

@author: Tapsas Apostolos
"""

import pandas as pd
import numpy as np

# Input Snips
snips = pd.read_table('20snps_file.txt', delim_whitespace=True,header = None)

# Input allele file
allels = pd.read_excel('snps_and_effect_allels.xlsx')
allels=allels['Effect Allele']

# Converting from Dataframe to numpy matrix
snips= snips.values

# Delete fisrt 6 collumns
snips= np.delete(snips,np.s_[0,1,2,3,4,5],axis=1)

# Code the allels to numbers
for i in range(0,len(snips)):
    k=0
    for j in range(0,len(snips[0,:])-1,2):
        if snips[i][j]=='0' and snips[i][j+1]=='0':
            snips[i][j]='miss'
        elif snips[i][j]==allels[k] and snips[i][j+1]==allels[k]:
            snips[i][j]=2
        elif snips[i][j]==allels[k] or snips[i][j+1]==allels[k]:
            snips[i][j]=1
        else:
            snips[i][j]=0
        k=k+1

# Save Data
#x=pd.DataFrame(snips)
#x.to_excel("snips_in_nums.xlsx")       

        
