# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:45:40 2020

@author: Szafran
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

data = pd.read_csv('train.csv')

#train-test split, try to stratify it later on (Pclass?)
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
test_X = test_set.drop('Survived', axis=1)
test_y = test_set['Survived'].copy()


#
train_set['fare_bins'] = pd.cut(train_set['Fare'], bins=[-1,8,15,45,1000], 
                                labels=['cheapest', 'economy', 'business', 'exclusive'])

plt.hist(train_set['fare_bins']) 
plt.show()#groups are (approximately) equally distributed

plt.scatter(train_set.groupby('fare_bins')['fare_bins'].max(), 
            train_set.groupby('fare_bins')['Survived'].mean())
plt.xlabel('Ticket price')
plt.ylabel('Survival rate')
plt.show() #strong correlation between ticket class and the survival rate

train_set['Age_groups'] = pd.cut(train_set['Fare'], bins=[0,13,17,35,65,95], 
                                labels=['kids', 'teens', 'young_adults', 'grown_ups', 'older_folks'])
plt.scatter(train_set.groupby('Age_groups')['Age_groups'].max(), 
            train_set.groupby('Age_groups')['Survived'].mean())
plt.xlabel('Age_group')
plt.ylabel('Survival rate')

