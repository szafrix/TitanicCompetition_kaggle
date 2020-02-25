# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:22:40 2020

@author: Szafran
"""
import numpy as np
import pandas as pd


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