# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:13:19 2020

@author: Szafran
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataPreparation as dp

data = pd.read_csv('train.csv')
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
test_X = test_set.drop('Survived', axis=1)
test_y = test_set['Survived'].copy()


#time for data manipulation
features = dp.dataPrepare(train_set.drop('Survived', axis=1)) #used module dataPreparation
attributes = train_set['Survived'].copy()


#its time for model.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier()
grid = [{'n_estimators':[3,10,30,35,40],
         'max_features':[2,3,4,5,6],
         'bootstrap':[False, True],
         'n_estimators':[2,3,10,15,20],
         'max_depth':[1,2,3],
         'criterion':['gini', 'entropy']}]
model = GridSearchCV(rf, grid, scoring='accuracy', cv=5)
model.fit(features, attributes)

#mierzenie wydajnosci modelu
from sklearn.model_selection import cross_val_score
mean_score = cross_val_score(model, features, attributes, scoring='accuracy').mean()
print(f"Mean score of cross validation using accuracy method: {mean_score}")


#analiza macierzy pomyłek
from sklearn.model_selection import cross_val_predict
cv_predictions = cross_val_predict(model, features, attributes)
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(attributes, cv_predictions)
precision = conf_matrix[1][1]/(conf_matrix[1][1]+conf_matrix[0][1])
recall = conf_matrix[1][1]/(conf_matrix[1][1]+conf_matrix[1][0])
print(f"Out of {conf_matrix[1][1]+conf_matrix[0][1]} people classified as 'Survived', {conf_matrix[1][1]} of them ({round(precision*100, 2)}%) were actual survivors.")
print(f"Out of {conf_matrix[1][0]+conf_matrix[1][1]} real survivors, {conf_matrix[1][1]} of them ({round(recall*100, 2)}%) were correctly classified as survivors.")

#kaggle upload
test_set = pd.read_csv('test.csv')
test_ready = dp.dataPrepare(test_set)
predictions = model.predict(test_ready)
file_to_submit = pd.DataFrame(test_set['PassengerId'])
file_to_submit['Survived'] = predictions

file_to_submit.to_csv('submissionRF.csv', index=False)