# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 10:58:13 2019

@author: arunbh
"""

import pandas as pd
from IPython.display import display
from IPython.display import Image

raw_data = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'], 
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
df_a = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
print(df_a)

raw_data = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'], 
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
df_b = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
print(df_b)

raw_data = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
df_n = pd.DataFrame(raw_data, columns = ['subject_id','test_id'])
print(df_n)

#Join the two dataframes along rows
df_new = pd.concat([df_a, df_b])
print('-----------------newconcat----------------')
print(df_new)

#Join the two dataframes along columns
#print(pd.concat([df_a, df_b], axis=1))

#Merge two dataframes along the subject_id value
print(pd.merge(df_new, df_n, on='subject_id'))
'''
#Merge two dataframes with both the left and right dataframes using the subject_id key
print(pd.merge(df_new, df_n, left_on='subject_id', right_on='subject_id'))

print(pd.merge(df_a, df_b, on='subject_id', how='outer'))

print(pd.merge(df_a, df_b, on='subject_id', how='inner'))

print(pd.merge(df_a, df_b, on='subject_id', how='right'))

print(pd.merge(df_a, df_b, on='subject_id', how='left'))

print(pd.merge(df_a, df_b, on='subject_id', how='left', suffixes=('_left', '_right')))

print(pd.merge(df_a, df_b, right_index=True, left_index=True))
'''
