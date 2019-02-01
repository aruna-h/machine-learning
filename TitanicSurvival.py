# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:33:53 2019

@author: arunbh
"""

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#getting data
train_df = pd.read_csv('D:/Users/arunbh/Downloads/AI-ML/titanic/dataset/train.csv')
test_df = pd.read_csv('D:/Users/arunbh/Downloads/AI-ML/titanic/dataset/test.csv')
label_df= pd.read_csv('D:/Users/arunbh/Downloads/AI-ML/titanic/dataset/gender_submission.csv')

#data exploration
train_df.info()
print("\n----------------train_df-------------------------\n")
print(train_df)
print("\n--------------test_df---------------------------\n")
print(test_df)

#Let’s take a more detailed look at what data is actually missing:
print("\n-----------missing data----------\n")
total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
print(missing_data.head(5))

#print(missing_data.describe())
print("column values are: ",train_df.columns.values)

#What features could contribute to a high survival rate ?
#1. Age and Sex:
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_df[train_df['Sex']=='female']
men = train_df[train_df['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
ax.set_title('Male')

#######################################################
print(train_df['Sex'].value_counts())
data = [train_df, test_df]
for dataset in data:
  dataset['Sex'].value_counts()
x=dataset.apply(pd.value_counts).fillna(0)
print('fgfjhghjhkjkjlk',x)
#############################################################
#3. Embarked, Pclass and Sex:
'''
FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()

#4. Pclass:
sns.barplot(x='Pclass', y='Survived', data=train_df)

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
'''
#5. SibSp and Parch:
print('---if relatives not alone=0(false), if relatives are alone=1---')
data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)

print(train_df['not_alone'].value_counts())

axes = sns.factorplot('relatives','Survived', 
                      data=train_df, aspect = 2.5)

#Data Preprocessing
#data preprocessing -First, I will drop ‘PassengerId’ from the train set, because it does not contribute to a persons survival probability. I will not drop it from the test set, since it is required there for the submission.
train_df = train_df.drop(['PassengerId'], axis=1)
print('------train_df after removing passenger id------',train_df)

#cabin
#A cabin number looks like ‘C123’ and the letter refers to the deck. Therefore we’re going to extract these and create a new feature, that contains a persons deck. Afterwords we will convert the feature into a numeric variable. The missing values will be converted to zero. In the picture below you can see the actual decks of the titanic, ranging from A to G.
import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
    
   # print(data)

# we can now drop the cabin feature
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)

data = [train_df, test_df]
print(data)

#age
for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)
print('checkiing age in train data is null :',train_df["Age"].isnull().sum())

print('mean of age data',mean)
print('std of age data',std)
#print('rand_age of age data',rand_age)

#embarked
print('describing embarked dataset:')
print(train_df['Embarked'].describe())

common_value = 'S'
data = [train_df, test_df]
print('after embarked replace:\n')
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
    #train_df.info()
data = [train_df, test_df]
print(data)

#fare
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

data = [train_df, test_df]
print(data)

#name
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

data = [train_df, test_df]
print(data)

#sex  
genders = {"male": 0, "female": 1}

data = [train_df, test_df]
print(data)
 
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
 
#ticket
train_df['Ticket'].describe()

train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)

#embarked
ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]
print(data)

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)

#Creating Categories:
#age    
data = [train_df, test_df]
#print(data)

for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

# let's see how it's distributed train_df['Age'].value_counts()
    #fare
    train_df.head(10)
    
data = [train_df, test_df]
print(data)

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    data = [train_df, test_df]
    #print(data)

#Creating new Features:
#age time class
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
    #print(data)

#fare per person
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
    #print(data)

# Let's take a last look at the training set, before we start training the models.
print(train_df.head(10))

#Building Machine Learning Models
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop(["PassengerId"], axis=1).copy()

decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train)  
Y_pred = decision_tree.predict(X_test)  #single data
print("Y predict",Y_pred)

# Predicting results for test dataset
submission = pd.DataFrame({
        "PassengerId": test_df['PassengerId'],
        "Survived": Y_pred
    })
#print("submission",submission.values)
Y_pred=submission["Survived"].values
Y_label=label_df["Survived"].values
    
#Y_label= test_df['Survived'].values
#acc_decision_tree2 = round(accuracy_score(label_df["Survived"].values, submission["Survived"].values) * 100,2)
acc_decision_tree2 = round(accuracy_score(Y_label, Y_pred) * 100,2)
print("accuracy2=",acc_decision_tree2)


#finding accuracy here
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100,2)
print("accuracy=",acc_decision_tree)

acc_decision_tree1 = round(decision_tree.score( X_test, Y_pred) * 100,2)
print("accuracy1=",acc_decision_tree1)


