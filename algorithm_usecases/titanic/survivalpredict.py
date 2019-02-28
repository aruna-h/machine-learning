# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:12:52 2019

@author: arunbh
"""

import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn import tree

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import GridSearchCV

# Loading the data
train = pd.read_csv('D:/Users/arunbh/Downloads/AI-ML/titanic/dataset/train.csv')
test = pd.read_csv('D:/Users/arunbh/Downloads/AI-ML/titanic/dataset/test.csv')
test_label = pd.read_csv('D:/Users/arunbh/Downloads/AI-ML/titanic/dataset/gender_submission.csv')


###########merging###############################
totalTest=pd.merge(test,test_label, on='PassengerId')
print(totalTest.shape)
print(totalTest)

totaldata=pd.concat([train,totalTest])
print(totaldata.shape)
print('totaldata------')
print(totaldata.head())
################################################

# Store our test passenger IDs for easy access
PassengerId = test['PassengerId']

full_data = [train, test]
#print(full_data)
print(train.head(3))

original_train = train.copy() # Using 'copy()' allows to clone the dataset, creating a different object with the same values

# Feature engineering steps taken from Sina and Anisotropic, with minor changes to avoid warnings
full_data = [train, test]

# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

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
'''
# Define function to extract titles from passenger names
def get_title(name):# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

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
    dataset.loc[np.isnan(dataset['Age'])
   
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
'''
for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    '''
    # Mapping titles
    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
'''
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
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
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)

print(train.head(3))
print(test.head(3))

####################survived passengers count###############################################

print('########### survived, not survived passengers ######################')
      
print('survived passengers in train data')
print(pd.value_counts(train['Survived'].values))

print('survived passengers in test data')
print(pd.value_counts(totalTest['Survived'].values))

print('total male and female in train data')
print(train['Sex'].value_counts())

print('total male and female in test data')
print(totalTest['Sex'].value_counts())

#Two step query to find sum of survived people, grouped by their sex(male1 and female0) in train
print('total male and female survived in train data')
out_survived = train.groupby(['Sex'])['Survived'].sum()
print(out_survived)

print('total male and female survived in test data')
#Two step query to find sum of survived people, grouped by their sex(male1 and female0) in test

out_survived = totalTest.groupby(['Sex'])['Survived'].sum()
print(out_survived)

print(train['Survived'].groupby(train['Sex']).mean())
print(totalTest['Survived'].groupby(totalTest['Sex']).mean())

#query to find sum of survived people, grouped by their passenger class (1 > 2 > 3)
print('survived people, grouped by their passenger class (1 > 2 > 3) in train data')
out_survived = train.groupby(['Pclass'])['Survived'].sum()
print(out_survived)

print('survived people, grouped by their passenger class (1 > 2 > 3) in test data')
out_survived = totalTest.groupby(['Pclass'])['Survived'].sum()
print(out_survived)

####################### pearson correlation model #######################################

#the relationship between our variables by plotting the Pearson Correlation between all the attributes in our dataset 
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

##############cross validation using k fold###############################
cv = KFold(n_splits=10)            # Desired number of Cross Validation folds
accuracies = list()
max_attributes = len(list(test))
depth_range = range(1, max_attributes + 1)

# Testing max_depths from 1 to max attributes
# Uncomment prints for details about each Cross Validation pass
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(max_depth = depth)
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

##############################################################################
'''
#print("Columns after preprocessing: ",full_data.columns)
# Applying these two columns to string type so that we can one hot encode it.
full_data['Sex'] = full_data['Sex'].apply(str)
full_data['IsAlone'] = full_data['IsAlone'].apply(str)
full_data['Has_Cabin'] = full_data['Has_Cabin'].apply(str)

########################## One Hot Encoding of Categorical features ################################
  
full_data_dummies=pd.get_dummies(full_data)
#print("full_dataset \n",full_dataset)
print("Columns after One Hot Encoding ",full_data_dummies.columns)
X = full_data_dummies.drop(['Survived'], axis=1).values 
y = full_data_dummies['Survived'].values

######################## grid search ##########################################################

print('######### grid search on decision tree#################')
      
param_grid = {"max_depth": [3,4,5,6,7,8,9],
              "min_samples_leaf": [1,2,3,4,5,6,7,8,9,10],
              "criterion":["gini", "entropy"]}
decision_tree = tree.DecisionTreeClassifier()
tree_cv = GridSearchCV(estimator = decision_tree,param_grid = param_grid, cv=3, n_jobs=-1)
tree_cv.fit(x_train, y_train)
print('hiiiiiiiiiiiiii')
print("tuned decision tree parameters: ",format(tree_cv.best_params_))
print("Best Score is: ",format(tree_cv.best_score_))

# Predicting results for test dataset
y_pred = tree_cv.predict(x_test)
submission1 = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_pred
    })
submission1.to_csv('submission1.csv', index=False)
acc_decision_tree_train = round(tree_cv.score(x_train, y_train) * 100, 2)
print("Accuracy on train dataset",acc_decision_tree_train)
acc_decision_tree_test = round(accuracy_score(test_label['Survived'].values , submission1['Survived'].values)*100, 2)
print("Accuracy on test dataset",acc_decision_tree_test)
'''
##################### fitting model #############################################

# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
y_train = train['Survived']
x_train = train.drop(['Survived'], axis=1).values 
x_test = test.values
###finding accuracy using  direct max_depth=3##################################
# Create Decision Tree with max_depth = 3
decision_tree = tree.DecisionTreeClassifier(max_depth = 3)
decision_tree.fit(x_train, y_train)

# Predicting results for test dataset
y_pred = decision_tree.predict(x_test)
submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
print('acc_decision_tree of train data',acc_decision_tree)
acc_decision_tree_test = round(accuracy_score(test_label['Survived'].values , submission['Survived'].values)*100, 2)
print("Accuracy on test dataset",acc_decision_tree_test)

##################### confusion matrix on train data #######################################

print('confusion matrix of train data')
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(decision_tree, x_train, y_train)
print(confusion_matrix(y_train, predictions))

from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_train, predictions))
print("Recall:",recall_score(y_train, predictions))

from sklearn.metrics import f1_score
print('f_score',f1_score(y_train, predictions))
'''
from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = decision_tree.predict_proba(x_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()
'''
#################### confusion matrix on test data ##########################

print('confusion matrix of test data')
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(decision_tree,x_test,test_label['Survived'].values)
print(confusion_matrix(test_label['Survived'].values, predictions))

from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(test_label['Survived'].values, predictions))
print("Recall:",recall_score(test_label['Survived'].values, predictions))

from sklearn.metrics import f1_score
print('f_score',f1_score(test_label['Survived'].values, predictions))

################creating tree diagram######################################

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
#note: dot tree1.dot -Tpng -o tree1.png        
    
#############################################################################



