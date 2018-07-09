# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 15:12:35 2018

@author: DANIEL MARTINEZ
"""

import EDA_framework as EDA #Framework with EDA helpful functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#--- Part 1: Data Preparation

#Load data
df = pd.read_csv('train.csv')
df.describe()

#-- Missing values --
#Look if any missing values
EDA.get_missing_data_table(df)

#Cabin is impossible to fix, for Embarked we delete the null observations, Age need a prediction
df = df.drop('Cabin', axis='columns')
df = EDA.delete_null_observations(df, column='Embarked')
df = df.reset_index(drop=True) #Reset indexes
df['Age'] = df['Age'].fillna(value=1000)

#--Feature importance analysis--
# family size = sibsp + parch
df['Family'] = df['SibSp'] + df['Parch']
df = df.drop('SibSp', axis='columns')
df = df.drop('Parch', axis='columns')

#Get titles and create group 'other'
name_row = df['Name'].copy()
name_row = pd.DataFrame(name_row.str.split(', ',1).tolist(), columns = ['Last name', 'Name'])
name_row = name_row['Name'].copy()
name_row = pd.DataFrame(name_row.str.split('. ',1).tolist(),columns=["Title","Name"])
name_row = name_row['Title'].copy()

titles = name_row.tolist()
for i in range(len(titles)):
    title = titles[i]
    if title != 'Master' and title != 'Miss' and title != 'Mr' and title !='Mrs':
        titles[i] = 'Other'

name_row = pd.DataFrame(titles, columns=['Title'])
df['Title'] = name_row.copy()

#Change Age based on statistics
test_df = df.copy()
test_df = pd.DataFrame([df['Age'].tolist(), df['Title'].tolist()]).transpose()
test_df.columns = ['Age','Title']

test_df_list = test_df.values
for i in range(len(test_df_list)):
    age = test_df_list[i][0]
    title = test_df_list[i][1]
    
    if age == 1000:
        if title == 'Master':
            test_df_list[i][0] = 5.19
        elif title == 'Miss':
            test_df_list[i][0] = 21.87
        elif title == 'Mr':
            test_df_list[i][0] = 32.18
        elif title == 'Mrs':
            test_df_list[i][0] = 35.48
        else:
            test_df_list[i][0] = 42.81

df['Age'] = test_df['Age'].copy()

#Drop not statisticak meaningful columns
df = df.drop('Name', axis='columns')
df = df.drop('Ticket', axis='columns')
df = df.drop('PassengerId', axis='columns')

#Treat categorical features
df['Age'] = df['Age'].astype('float64')
df = EDA.transform_dummy_variables(df,['Sex','Pclass','Embarked','Title'])

#Getting X and y
X_train = df.iloc[:,1:].values
y = df.iloc[:,0].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

#-- Part 2: Modeling

#Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

folds = 4
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y), verbose=3, random_state=1001 )
random_search.fit(X_train, y)

xgboost_classifier = random_search.best_estimator_

#Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
params = {
        'n_estimators': [5, 10, 15],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'max_depth': [None, 3, 4, 5]
        }

folds = 4
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y), verbose=3, random_state=1001 )
random_search.fit(X_train, y)

randomforest_classifier = random_search.best_estimator_

#Fitting SVC to the Training set
from sklearn.svm import SVC
classifier = SVC(probability=True)
classifier.fit(X_train, y)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
params = {
        'C': [0.5, 1, 1.5],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'gamma': [0.001, 0.0001],
        'class_weight': [None, 'balanced']
        }

folds = 4
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y), verbose=3, random_state=1001 )
random_search.fit(X_train, y)

svc_classifier = random_search.best_estimator_

#Ensemble 
from sklearn.ensemble import VotingClassifier
classifier = VotingClassifier(estimators=[('xgb', xgboost_classifier), ('rf',randomforest_classifier), ('svc',svc_classifier)], voting='soft')
classifier.fit(X_train, y)

#Cross-validation score
from sklearn.model_selection import cross_val_score
accuaricies = cross_val_score(estimator=classifier, X=X_train, y=y, cv=5)
print(accuaricies.mean())

#-- Part 3: Predicting test dataset
#Importing test dataset
df_test = pd.read_csv('test.csv')
df_test.describe()

#Features transformation
EDA.get_missing_data_table(df_test)
df_test = EDA.imput_nan_values(df_test,'Fare','median')
df_test = df_test.drop('Cabin', axis='columns')
df_test['Age'] = df_test['Age'].fillna(value=1000)

df_test['Family'] = df_test['SibSp'] + df_test['Parch']
df_test = df_test.drop('SibSp', axis='columns')
df_test = df_test.drop('Parch', axis='columns')

name_row = df_test['Name'].copy()
name_row = pd.DataFrame(name_row.str.split(', ',1).tolist(), columns = ['Last name', 'Name'])
name_row = name_row['Name'].copy()
name_row = pd.DataFrame(name_row.str.split('. ',1).tolist(),columns=["Title","Name"])
name_row = name_row['Title'].copy()

titles = name_row.tolist()
for i in range(len(titles)):
    title = titles[i]
    if title != 'Master' and title != 'Miss' and title != 'Mr' and title !='Mrs':
        titles[i] = 'Other'

name_row = pd.DataFrame(titles, columns=['Title'])
df_test['Title'] = name_row.copy()

test_df = df_test.copy()
test_df = pd.DataFrame([df_test['Age'].tolist(), df_test['Title'].tolist()]).transpose()
test_df.columns = ['Age','Title']

test_df_list = test_df.values
for i in range(len(test_df_list)):
    age = test_df_list[i][0]
    title = test_df_list[i][1]
    
    if age == 1000:
        if title == 'Master':
            test_df_list[i][0] = 5.19
        elif title == 'Miss':
            test_df_list[i][0] = 21.87
        elif title == 'Mr':
            test_df_list[i][0] = 32.18
        elif title == 'Mrs':
            test_df_list[i][0] = 35.48
        else:
            test_df_list[i][0] = 42.81

df_test['Age'] = test_df['Age'].copy()

df_test = df_test.drop('Name', axis='columns')
df_test = df_test.drop('Ticket', axis='columns')
df_test = df_test.drop('PassengerId', axis='columns')

df_test['Age'] = df_test['Age'].astype('float64')
df_test = EDA.transform_dummy_variables(df_test,['Sex','Pclass','Embarked','Title'])

#Predictions
X_test = df_test.values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X_test)

pred = classifier.predict(X_test)

# Create result dataframe
test_dataset = pd.read_csv('test.csv')
ps_id = test_dataset.iloc[:,0].values
d = {'PassengerId':ps_id, 'Survived':pred}
df = pd.DataFrame(data=d)
df = df.set_index('PassengerId')
df.to_csv('predictions.csv')