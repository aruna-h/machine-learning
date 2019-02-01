# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:12:20 2019

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

totalTest=pd.merge(test,test_label, on='PassengerId')
print(totalTest.shape)
print(totalTest)

totaldata=pd.concat([train,totalTest])
print(totaldata.shape)
print('totaldata------',totaldata)
print(totaldata['PassengerId'])