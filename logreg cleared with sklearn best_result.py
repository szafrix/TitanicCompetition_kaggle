# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:49:56 2020

@author: Szafran
"""

"""
logistic regression model
train set splitted for measurement - not all data used for fitting
added variable relatives
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('train.csv')

#train-test split, try to stratify it later on (Pclass?)
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
test_X = test_set.drop('Survived', axis=1)
test_y = test_set['Survived'].copy()

#time for data manipulation
features = train_set.drop('Survived', axis=1)
attributes = train_set['Survived'].copy()

def dataPrepare(features):
    #let's drop Name, Cabin, PassengerId and Ticket (is Ticket holding any info?)
    features = features.drop(['Name', 'Cabin', 'PassengerId', 'Ticket', 'Fare', 'Embarked'], axis=1)
    
    #Missing values - we only have them in Age feature, lets fill them with median
    features['Age'] = features['Age'].where(lambda x: x<100, features['Age'].median())
    #Relatives
    def relatives_summarizer(rels):
        if rels ==0:
            rels = "Solo"
        elif rels in range (1,4):
            rels = 'Few_relatives'
        else:
            rels = 'Many_relatives'
        return rels
    
    features['Relatives'] = features['Parch'] + features['SibSp']
    features['Relatives'] = features['Relatives'].apply(lambda x: relatives_summarizer(x))
        
    #categorical variables
    features_cat = pd.get_dummies(features[['Pclass', 'Sex', 'Relatives']].astype(str).copy())
    
    #standarization
    features_num_before = features.drop(['Pclass', 'Sex', 'Relatives'], axis=1)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_num = pd.DataFrame(scaler.fit_transform(features_num_before), columns=features_num_before.columns, index=features.index)
    
    #merge
    features_final = pd.concat([features_cat, features_num], axis=1)
    features_final = features_final.drop('Relatives_Few_relatives', axis=1)
    return features_final

#its time for model.
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(dataPrepare(features), attributes)
logreg.predict(dataPrepare(test_X))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(dataPrepare(test_X), test_y)))

test_set = pd.read_csv('test.csv')
test_ready = dataPrepare(test_set)

predictions = logreg.predict(test_ready)
file_to_submit = pd.DataFrame(test_set['PassengerId'])
file_to_submit['Survived'] = predictions

file_to_submit.to_csv('submissionLOGREG.csv', index=False)