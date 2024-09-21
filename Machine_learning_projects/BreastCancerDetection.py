import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import csv

linebreak='----------------------------------------------------------------------------------------------------'

ds = pd.read_csv('./dataset/breastCancerDataset.csv', delimiter=',')

# --------------------------------------------------------------------------------------------------------------
# Reviewing the data to see what is available in the dataset
# --------------------------------------------------------------------------------------------------------------
print(ds.head())
ds.shape
ds.info()

print(linebreak)

columns = ds.select_dtypes(include='object').columns
print('Column headers that are formatted as objects: ' + str(columns))
print('Number of columns that are objects: ' + str(len(columns)) + '\n')

numbers = ds.select_dtypes(include=['float64','int64']).columns
print('Column headers that are formatted as float64 or int64: ' + str(numbers))
print('Number of columns that are float64 or int64: '+str(len(numbers)) + '\n')

print('Dataset Description:\n'+str(ds.describe()))

print(linebreak)

# --------------------------------------------------------------------------------------------------------------
# Dealing with the missing values in this dataset
# --------------------------------------------------------------------------------------------------------------
if ds.isnull().values.any():
    print('There are ' + str(ds.isnull().values.sum()) + ' NULL values in dataset')
    print('There are ' + str(len(ds.columns[ds.isnull().any()])) + ' column(s) with NULL values')
    print('List of columns with NULL values: ' + str(ds.columns[ds.isnull().any()].values))
    # Drop columns with no values in them.
    drop_cols = []
    for col in ds.columns[ds.isnull().any()].values :
        if ds[str(col)].count() < 1:
            drop_cols.append(col)
    ds.drop(drop_cols, axis = 1, inplace=True)
    ds.shape

    print('Columns with all NULL values have been dropped.')
    print('There are ' + str(ds.isnull().values.sum()) + ' NULL values in dataset')

print(linebreak)

# --------------------------------------------------------------------------------------------------------------
# Categorical Data
# --------------------------------------------------------------------------------------------------------------
print(ds.head())
object_cols=ds.select_dtypes(include='object').columns

# In this example I want to split the data by diagnosis.
print(object_cols.values)
print(ds['diagnosis'].unique())
print(ds['diagnosis'].nunique())

object_cols.values
ds['diagnosis'].unique()
ds['diagnosis'].nunique()

# I want to split that data into 2
ds = pd.get_dummies(data=ds, dtype=int, drop_first=True)
# ds = ds.rename(columns={'' : 'diagnosis'})
# Drop first removes the diagnosis column

print(ds.head())

print(linebreak)

# --------------------------------------------------------------------------------------------------------------
# Countplot graphs
# --------------------------------------------------------------------------------------------------------------
sns.countplot(ds, x='diagnosis_M', label='Count')
plt.show()

print('\nBinine diagnosis count: ' + str((ds.diagnosis_M == 0).sum()))
print('Malignant diagnosis count: ' + str((ds.diagnosis_M == 1).sum()))

print(linebreak)

# --------------------------------------------------------------------------------------------------------------
# Correlation Matrix and Heat Map
# --------------------------------------------------------------------------------------------------------------
ds_2 = ds.drop(columns='diagnosis_M')
print(ds_2.head())

ds_2.corrwith(ds['diagnosis_M']).plot.bar(
    figsize=(20,10), title = 'Correclated with diagnosis', rot=45, grid=True
)
plt.show()

corr = ds.corr()
print(corr)

#HeatMap
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True)
plt.show()

# --------------------------------------------------------------------------------------------------------------
# 
# --------------------------------------------------------------------------------------------------------------