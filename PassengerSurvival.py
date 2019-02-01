# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 12:59:39 2019

@author: arunbh
"""

# Imports needed for the script
import numpy as np
import pandas as pd
import re
#import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn import tree
import plotly.offline as py
py.init_notebook_mode(connected=True)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Loading the data
train = pd.read_csv('D:/Users/arunbh/Downloads/AI-ML/dataset/train.csv')
test = pd.read_csv('D:/Users/arunbh/Downloads/AI-ML/dataset/test.csv')
label=pd.read_csv('D:/Users/arunbh/Downloads/AI-ML/dataset/gender_submission.csv')

# Store our test passenger IDs for easy access
PassengerId = test['PassengerId']

# Showing overview of the train dataset
print(train.head(3))

# Copy original dataset in case we need it later when digging into interesting features
# WARNING: Beware of actually copying the dataframe instead of just referencing it
# "original_train = train" will create a reference to the train variable (changes in 'train' will apply to 'original_train')
original_train = train.copy() # Using 'copy()' allows to clone the dataset, creating a different object with the same values

# Feature engineering steps taken from Sina and Anisotropic, with minor changes to avoid warnings
full_data = [train, test]

# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
print(full_data)

# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

# Remove all NULLS in the Age column
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # Next line has been improved to avoid warning
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
    
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
     # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
  
    # Mapping titles
    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Mapping Embarked
#dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
# Mapping Fare
dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
dataset['Fare'] = dataset['Fare'].astype(int)
    
# Mapping Age
dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
dataset.loc[ dataset['Age'] > 64, 'Age'] ;
   
# Feature selection: remove variables no longer containing relevant information
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)

print(train.head(3))

# Applying these two columns to string type so that we can one hot encode it.
train['IsAlone'] = train['IsAlone'].apply(str)
test['IsAlone'] = test['IsAlone'].apply(str)
train['Has_Cabin'] = train['Has_Cabin'].apply(str)
test['Has_Cabin'] = test['Has_Cabin'].apply(str)

########################## One Hot Encoding of Categorical features #################

train_dummies=pd.get_dummies(train)
print("Final columns in train dataset \n", train_dummies.columns)
test_dummies=pd.get_dummies(test)
print("Final columns in test dataset \n", test_dummies.columns)

# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models'''
x_train = train_dummies.drop(['Survived'], axis=1).values 
y_train = train_dummies['Survived']
x_test = test_dummies.values


print('----------------------------------------------------------------')

colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

train[['Title', 'Survived']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum'])
# Since "Survived" is a binary class (0 or 1), these metrics grouped by the Title feature represent:
    # MEAN: survival rate
    # COUNT: total observations
    # SUM: people survived

# title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 
    
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).agg(['mean', 'count', 'sum'])
# Since Survived is a binary feature, this metrics grouped by the Sex feature represent:
    # MEAN: survival rate
    # COUNT: total observations
    # SUM: people survived
    
# sex_mapping = {{'female': 0, 'male': 1}} 
    
    # Let's use our 'original_train' dataframe to check the sex distribution for each title.
# We use copy() again to prevent modifications in out original_train dataset
title_and_sex = original_train.copy()[['Name', 'Sex']]

# Create 'Title' feature
title_and_sex['Title'] = title_and_sex['Name'].apply(get_title)

# Map 'Sex' as binary feature
title_and_sex['Sex'] = title_and_sex['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Table with 'Sex' distribution grouped by 'Title'
title_and_sex[['Title', 'Sex']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum'])


'''
# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
y_train = train['Survived']
x_train = train.drop(['Survived'], axis=1).values 
x_test = test.values
'''
######################cross validation######################################

#Finding best tree depth with the help of Cross Validation
cv = KFold(n_splits=10)            # Desired number of Cross Validation folds
accuracies = list()
max_attributes = len(list(test))
depth_range = range(1, max_attributes + 1)

print("cv -",cv)

# Testing max_depths from 1 to max attributes
# Uncomment prints for details about each Cross Validation pass
for depth in depth_range:
    fold_accuracy = []
    tree_model = DecisionTreeClassifier(max_depth = depth)
    # print("Current max depth: ", depth, "\n")
    for train_fold, valid_fold in cv.split(train):
        f_train = train.loc[train_fold] # Extract train data with cv indices
        f_valid = train.loc[valid_fold] # Extract valid data with cv indices

        model = tree_model.fit(X = f_train.drop(['Survived'], axis=1), 
                               y = f_train["Survived"]) # We fit the model with the fold train data
        valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1), 
                                y = f_valid["Survived"])# We calculate accuracy with the fold validation data
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)
    # print("Accuracy per fold: ", fold_accuracy, "\n")
    # print("Average accuracy: ", avg)
    # print("\n")
    
# Just to show results conveniently
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))

############################grid search#################################
from sklearn.grid_search import GridSearchCV
#define parameter values that to be searched
param_grid = {"max_depth": [1,2,3,4,5,6,7,8,9,10],
              "min_samples_leaf": [6,7,8,9,10,11,12,13,14],
              "criterion":["gini", "entropy"]}
decision_tree = tree.DecisionTreeClassifier()
decisiontree_grid = GridSearchCV(estimator = decision_tree,param_grid = param_grid, verbose=False,  cv=3, n_jobs=-1)
decisiontree_grid .fit(x_train, y_train)
print("tuned decision tree parameters: ",format(decisiontree_grid .best_params_))
print("Best Score is: ",format(decisiontree_grid .best_score_))


#Final Tree

# Create Decision Tree with max_depth = 3

#decision_tree.fit(x_train, y_train)

# Predicting results for test dataset
y_pred = decision_tree.predict(x_test)

submission1 = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_pred
    })
submission1.to_csv('submission1.csv', index=False)

acc_decision_tree_train = round(decisiontree_grid.score(x_train, y_train) * 100, 2)
print("Accuracy on train dataset",acc_decision_tree_train)
acc_decision_tree_test = round(accuracy_score(label['Survived'].values , submission1['Survived'].values)*100, 2)
print("Accuracy on test dataset",acc_decision_tree_test)


# Export our trained model as a .dot file
with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 3,
                              impurity = True,
                              feature_names = list(train.drop(['Survived'], axis=1)),
                              class_names = ['Died', 'Survived'],
                              rounded = True,
                              filled= True )
#dot tree1.dot -Tpng -o tree1.png :this is command to convert dot file into png file


   #finding accuracy here
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100,2)
print("acc of training data=",acc_decision_tree)

y_label=label["Survived"].values
y_pred=submission1["Survived"].values

acc_decision_tree1 = round(accuracy_score(y_label, y_pred) * 100, 2)
print("acc of survived data=",acc_decision_tree1)

