# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:54:39 2019

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

#from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# Loading the data
train = pd.read_csv('D:/Users/arunbh/Downloads/AI-ML/algorithm_usecases/titanic/dataset/train.csv')
test = pd.read_csv('D:/Users/arunbh/Downloads/AI-ML/algorithm_usecases/titanic/dataset/test.csv')
test_label = pd.read_csv('D:/Users/arunbh/Downloads/AI-ML/algorithm_usecases/titanic/dataset/gender_submission.csv')

###########    merging   ###############################
totalTest=pd.merge(test,test_label, on='PassengerId')
print(totalTest.shape)
print(totalTest)

totaldata=pd.concat([train,totalTest])
print(totaldata.shape)
print('totaldata------')
print(totaldata.head())
print("totaldata.columns ",totaldata.columns)


########################### data preprocessing #####################################
# Store our test passenger IDs for easy access
PassengerId = totaldata['PassengerId']

# Feature engineering steps taken from Sina and Anisotropic, with minor changes to avoid warnings
# Feature that tells whether a passenger had a cabin on the Titanic
totaldata['Has_Cabin'] = totaldata["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
totaldata['Has_Cabin'] = totaldata["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
print(totaldata.columns)

# Create new feature FamilySize as a combination of SibSp and Parch
totaldata['FamilySize'] = totaldata['SibSp'] + totaldata['Parch'] + 1
print(totaldata.columns)

# Create new feature IsAlone from FamilySize
totaldata['IsAlone'] = 0
totaldata.loc[totaldata['FamilySize'] == 1, 'IsAlone'] = 1
print(totaldata.columns)

# Remove all NULLS in the Embarked column
totaldata['Embarked'] = totaldata['Embarked'].fillna('S')

# Remove all NULLS in the Fare column
totaldata['Fare'] = totaldata['Fare'].fillna(train['Fare'].median())

# Remove all NULLS in the Age column
mean = totaldata["Age"].mean()
std = totaldata["Age"].std()
is_null = totaldata["Age"].isnull().sum()
# compute random numbers between the mean, std and is_null
rand_age = np.random.randint(mean - std, mean + std, size = is_null)
# fill NaN values in Age column with random values generated
age_slice = totaldata["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age
totaldata["Age"] = age_slice
totaldata["Age"] = totaldata["Age"].astype(int)
totaldata["Age"].isnull().sum()

# Mapping Sex
totaldata['Sex'] = totaldata['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Mapping Embarked
totaldata['Embarked'] = totaldata['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
# Mapping Fare
totaldata.loc[ totaldata['Fare'] <= 7, 'Fare'] 						        = 0
totaldata.loc[(totaldata['Fare'] > 7) & (totaldata['Fare'] <= 14), 'Fare'] = 1
totaldata.loc[(totaldata['Fare'] > 14) & (totaldata['Fare'] <= 31), 'Fare']   = 2
totaldata.loc[ totaldata['Fare'] > 31, 'Fare'] 	       					        = 3
totaldata['Fare'] = totaldata['Fare'].astype(int) 
    
# Mapping Age
totaldata.loc[ totaldata['Age'] <= 16, 'Age'] 					       = 0
totaldata.loc[(totaldata['Age'] > 16) & (totaldata['Age'] <= 32), 'Age'] = 1
totaldata.loc[(totaldata['Age'] > 32) & (totaldata['Age'] <= 48), 'Age'] = 2
totaldata.loc[(totaldata['Age'] > 48) & (totaldata['Age'] <= 64), 'Age'] = 3
totaldata.loc[ totaldata['Age'] > 64, 'Age'] = 4
totaldata['Age'] = totaldata['Age'].astype(int) 
     
# Feature selection: remove variables no longer containing relevant information
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch']
totaldata=totaldata.drop(drop_elements, axis = 1)
print(totaldata.columns)

#totaldata = totaldata.drop(["Title"])

#print(totaldata.columns)
print("Columns after preprocessing: ",totaldata.columns)

# Applying these two columns to string type so that we can one hot encode it.
totaldata['Sex'] = totaldata['Sex'].apply(str)
totaldata['IsAlone'] = totaldata['IsAlone'].apply(str)
totaldata['Has_Cabin'] = totaldata['Has_Cabin'].apply(str)

########################## One Hot Encoding of Categorical features ################################

totaldata_dummies=pd.get_dummies(totaldata)
#print("full_dataset \n",full_dataset)
print("\n")
print("Columns after One Hot Encoding ",totaldata_dummies.columns)

####################### pearson correlation model #######################################

#the relationship between our variables by plotting the Pearson Correlation between all the attributes in our dataset 
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
#plt.title('Pearson Correlation of Features', y=1.05, size=15)
#sns.heatmap(totaldata.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

plt.title('spearman Correlation of Features', y=1.05, size=15)
sns.heatmap(totaldata.astype(float).corr(method ='spearman'),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

############## K-fold cross validation ###############################
max_attributes = len(list(totaldata))
depth_range = range(1, max_attributes + 1)
accuracies = []
for depth in depth_range:
     random_forest = RandomForestClassifier(max_depth = depth)
     scores = cross_val_score(random_forest, totaldata.drop(["Survived"], axis=1).values, totaldata[["Survived"]].values, cv=3 , scoring='accuracy')
     accuracies.append(scores.mean())
# Just to show results conveniently
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print("\n")
print(df.to_string(index=False))
    # print("Accuracy per fold: ", fold_accuracy, "\n")
    # print("Average accuracy: ", avg)
    # print("\n") 

######################### splitting data into 50:50 ####################################

X = totaldata_dummies.drop(['Survived'], axis=1).values 
Y = totaldata_dummies['Survived'].values

x_train, x_test, y_train, y_test = train_test_split(X, Y,test_size=0.20,random_state=42)
print("\nx_train:\n")
print(x_train)
print(x_train.shape)

print("\nx_test:\n")
print(x_test)
print(x_test.shape)

print("\ny_train:\n")
print(y_train)
print(y_train.shape)

print("\ny_test:\n")
print(y_test)
print(y_test.shape)

###################### survival of passengers with respect to features #################
print('xtrain',x_train.shape)
print("type(y_train): \n",type(y_train))
y_train=y_train[:, None]
print("y_train: \n",y_train.shape)
df_train = np.concatenate((x_train,y_train),axis=1)
print('aaaaaaaaaaaaaaaaaaaa')
print("df_train: \n",df_train.shape)
df_train = pd.DataFrame.from_records(df_train, columns=['Age', 'Embarked', 'Fare', 'Pclass', 'FamilySize', 'Sex_0','Sex_1','Has_Cabin_0', 'Has_Cabin_1', 'IsAlone_0', 'IsAlone_1','Survived'])
print("\n")
print("df_train: \n",df_train.shape)

a = df_train.groupby('Survived').count()
print("\n")
print("Dead and Survived in df_train \n",a)

b = df_train.groupby(['Sex_0','Survived']).size()
print("\n")
print("Survival rate related to Sex in df_train dataset \n",b)

y_test=y_test[:, None]
df_Test = np.concatenate((x_test,y_test),axis=1)
df_Test = pd.DataFrame.from_records(df_Test, columns=['Age', 'Embarked', 'Fare', 'Pclass', 'FamilySize', 'Sex_0','Sex_1', 'Has_Cabin_0', 'Has_Cabin_1', 'IsAlone_0', 'IsAlone_1','Survived'])
print("\n")
print("df_Test: \n",df_Test.shape)

c = df_Test.groupby('Survived').count()
print("\n")
print("Dead and Survived in df_Test \n",c)

d = df_Test.groupby(['Sex_0','Survived']).size()
print("\n")
print("Survival rate related to Sex in df_Test dataset \n",d)

###################### fitting model ############################################
#Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
y_trainsplit = df_train['Survived']
x_trainsplit = df_train.drop(['Survived'], axis=1).values 
x_testsplit = x_test
y_testsplit = y_test

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(x_trainsplit, y_trainsplit)

Y_prediction = random_forest.predict(x_test)

random_forest.score(x_trainsplit, y_trainsplit)
acc_random_forest = round(random_forest.score(x_trainsplit, y_trainsplit) * 100, 2)
print('acc_random_forest of train dataset',acc_random_forest)
metrics.accuracy_score(y_testsplit , Y_prediction)
acc_random_forest_test = round(metrics.accuracy_score(y_testsplit , Y_prediction)*100, 2)
print("Accuracy on test dataset",acc_random_forest_test)

#metrics.roc_auc_score(y_testsplit , Y_prediction)
#print('acc',round(metrics.roc_auc_score(y_testsplit , Y_prediction)*100, 2))


######################## grid search ##########################################################
#random_forest = RandomForestClassifier()
param_grid ={"max_depth": [3],
              "min_samples_leaf": [1],
              "criterion":["gini", "entropy"]}
print("hiiiiii")
#from sklearn.model_selection import GridSearchCV, cross_val_score
rf = RandomForestClassifier(n_estimators=100,
                            #max_features='auto', 
                            oob_score=True, random_state=1, n_jobs=-1)
grid = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
grid.fit(x_trainsplit, y_trainsplit)
print("hiiiiii1")
print("grid",grid)
print(grid.best_score_)
print("hiiiiii2")
print("\n")
print("tuned decision tree parameters: ",format(grid.best_params_))
print("\n")
print("Best Score is: ",format(grid.best_score_))

# Predicting results for test dataset
y_train_pred = grid.predict(x_trainsplit)

acc_decision_tree_train = round(grid.score(x_trainsplit, y_trainsplit) * 100, 2)
print("\n")
print("Accuracy on df_train dataset(GS)",acc_decision_tree_train)

y_test_pred = grid.predict(x_testsplit)

acc_decision_tree_test = round(grid.score(x_testsplit, y_testsplit) * 100, 2)
print("\n")
print("Accuracy on test dataset(GS)",acc_decision_tree_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print("Confusion matrix of total dataset",confusion_matrix(y_train, y_train_pred))
print("Precision score of total dataset: ",precision_score(y_train, y_train_pred))
print("Recall score of total dataset: ", recall_score(y_train, y_train_pred))

print("Confusion matrix of test \n",confusion_matrix(y_test, y_test_pred))
print("Precision score of test: ",precision_score(y_test, y_test_pred))
print("Recall score of test: ", recall_score(y_test, y_test_pred))

##################### confusion matrix on train data #######################################

print('confusion matrix of train data')
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, x_trainsplit, y_trainsplit)
print(confusion_matrix(y_trainsplit, predictions))

from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_trainsplit, predictions))
print("Recall:",recall_score(y_trainsplit, predictions))

from sklearn.metrics import f1_score
print('f_score',f1_score(y_trainsplit, predictions))

##################### confusion matrix on test data #######################################

print('confusion matrix of test data')
predictions = cross_val_predict(random_forest,x_testsplit,y_testsplit)
print(confusion_matrix(y_testsplit, predictions))

from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_testsplit, predictions))
print("Recall:",recall_score(y_testsplit, predictions))

from sklearn.metrics import f1_score
print('f_score',f1_score(y_testsplit, predictions))

################creating tree diagram######################################

from sklearn.tree import export_graphviz
for x in range(0, 1):
    estimator = random_forest.estimators_[x]
# Export our trained model as a .dot file
    with open("tree"+str(x)+".dot", 'w') as f:
     f = export_graphviz(estimator,
                         out_file=f, 
                        # max_depth = 3,
                         impurity = True,
                         feature_names = list(df_train.drop(['Survived'], axis=1)),
                         class_names = ['Died', 'Survived'],
                         rounded = True,
                         filled = True)

#note: dot tree.dot -Tpng -o tree.png     
     
