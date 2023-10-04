# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 16:03:45 2023
@author: harisushehu
"""

import pandas as pd
import glob
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import StratifiedKFold
import statistics
from scipy.stats import pearsonr
import seaborn as sns

def ratioedData(test_y, test_X):
    countOnes = np.count_nonzero(test_y == 1)
    countZeros = np.count_nonzero(test_y == 0)

    if countOnes > countZeros:
        identifier = 1
    else:
        identifier = 0

    difference = abs(countZeros - countOnes)
    countRemoved = 0

    # Delete excess to make 1:1
    for i in range(0, len(test_y) - 1):
        if countRemoved == difference:
            break

        if ((i < len(test_y)) and (test_y[i] == identifier) and (countRemoved < difference)):
            test_y = np.delete(test_y, obj=i, axis=0)
            test_X = np.delete(test_X, obj=i, axis=0)
            countRemoved += 1

    return test_y, test_X

print("Reading data...")

path = '' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
    li.append(df)

dataset = pd.concat(li, axis=0, ignore_index=True)

# Replace NaN with 0
dataset = dataset.fillna(0)

print("Evaluating...")

X = dataset.iloc[:, df.columns != 'LABEL_SR_Arousal']
y = dataset.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

print("X is:", X.shape)
print("y is:", y.shape)

print("Reading data for preprocessing...")
path = '' # use your path 
all_files = sorted(glob.glob(path + "/*.csv"))

full_list = []

for normloop in range(0, 154): #154
    participant = normloop
    flag = False
    fileReader = []
    increament = 0
    for filename in all_files:
        if len(str(participant)) == 1:
            partNo = "00" + str(participant)
        elif len(str(participant)) == 2:
            partNo = "0" + str(participant)
        else:
            partNo = str(participant)
        if partNo in filename:
            reader = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
            lines = len(reader)
            increament = increament + lines
            fileReader.append(reader)
            flag = True
    if flag == True:
        print("Evaluating for participant " + str(normloop) + "...")
        full_list.extend(fileReader)
        test_reader = full_list
        full_list = []
        test_data = pd.concat(test_reader, axis=0, ignore_index=True)
        # Drop NA
        test_data = test_data.dropna()
        import numpy as np
        test_X = test_data.iloc[:, test_data.columns != 'LABEL_SR_Arousal']
        test_X = np.array(test_X)
        test_y = test_data.iloc[:, test_data.columns == 'LABEL_SR_Arousal'].values
        # Scale test
        scaler_X2 = StandardScaler()
        scaler_y2 = StandardScaler()
        # Normalize test X and y
        test_X = scaler_X2.fit_transform(test_X)
        test_y = scaler_y2.fit_transform(test_y)
    else:
        print("Participant does not exist")

Heart_rate = X['heartrate_mean'].values
GSR =  X['GSR_mean'].values

my_correlation = np.corrcoef(Heart_rate, GSR)

# Calculate Pearson's correlation
corr, _ = pearsonr(Heart_rate, GSR)
print('Pearson\'s correlation: %.3f' % corr)

sns.heatmap(my_correlation)

X_new = X[['heartrate_mean', 'GSR_mean', 'IRPleth_mean', 'Respir_mean']]
cormat = X_new.corr()
sns.heatmap(cormat)
