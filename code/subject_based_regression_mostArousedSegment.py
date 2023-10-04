# -*- coding: utf-8 -*-
"""
Created on Thu May 26 18:36:48 2022
@author: harisushehu
"""

import pandas as pd
import glob
import os
import csv
from csv import writer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy import sqrt


# Append data in CSV function
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

# Calculate NRMSE
def nrmse(rmse, y_test): 
    nrmse = (rmse) / ((max(y_test) - min(y_test)))
    return nrmse[0]

csvFileName = "../results/EMAP_subjectBased_regression_mostArousedSegment.csv"

# Read in CSV file
if os.path.exists(csvFileName):
    print()
else:
    with open(csvFileName, 'a+', newline='') as f:
        header = ['Iteration', 'LR-RMSE', 'DT-RMSE', 'LR-NRMSE', 'DT-NRMSE'] 
        filewriter = csv.DictWriter(f, fieldnames=header)
        filewriter.writeheader()

print("Reading data...")

path = '' #use your path
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
path = '' #use your path
all_files = sorted(glob.glob(path + "/*.csv"))

full_list = []

DT_rmse = []
Linear_rmse = []
NDT_nrmse = []
NLinear_nrmse = []

for normloop in range(0, 154):  # 154
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
            reader = reader.fillna(0)
            if reader['LABEL_SR_Arousal'].any() == True:
                maximum = max(reader['LABEL_SR_Arousal'])
                for i in range(0, len(reader)):
                    reader['LABEL_SR_Arousal'][i] = maximum
            fileReader.append(reader)
            flag = True
    if flag == True:
        print("Evaluating for participant " + str(normloop) + "...")
        full_list.extend(fileReader)
        test_reader = full_list
        full_list = []
        test_data = pd.concat(test_reader, axis=0, ignore_index=True)
        test_data = test_data.dropna()
        import numpy as np
        test_X = test_data.iloc[:, test_data.columns != 'LABEL_SR_Arousal']
        test_X = np.array(test_X)
        test_y = test_data.iloc[:, test_data.columns == 'LABEL_SR_Arousal'].values
        scaler_X2 = StandardScaler()
        scaler_y2 = StandardScaler()
        test_X = scaler_X2.fit_transform(test_X)
        test_y = scaler_y2.fit_transform(test_y)
        from sklearn.model_selection import KFold
        LR_list_rmse = []
        DT_list_rmse = []
        LR_list_nrmse = []
        DT_list_nrmse = []
        # Prepare cross-validation
        kfold = KFold(3)
        for train_index, test_index in kfold.split(test_X, test_y):
            X_train, X_test = test_X[train_index], test_X[test_index]
            y_train, y_test = test_y[train_index], test_y[test_index]
            LinearReg = LinearRegression().fit(X_train, y_train)
            DTReg = DecisionTreeRegressor(random_state=1).fit(X_train, y_train)
            LinearPred = LinearReg.predict(X_test)
            DTPred = DTReg.predict(X_test)
            Linear_mse = mean_squared_error(y_test, LinearPred)
            Linear_rmse = sqrt(Linear_mse)
            DT_mse = mean_squared_error(y_test, DTPred)
            DT_rmse = sqrt(DT_mse)
            if (max(y_test) - min(y_test)) == 0:
                NLinear_rmse = Linear_rmse / 1
                NDT_rmse = DT_rmse / 1
            else:
                NLinear_rmse = nrmse(Linear_rmse, y_test)
                NDT_rmse = nrmse(DT_rmse, y_test)
            LR_list_rmse.append(Linear_rmse)
            DT_list_rmse.append(DT_rmse)
            LR_list_nrmse.append(NLinear_rmse)
            DT_list_nrmse.append(NDT_rmse)
        NDT_nrmse.append(np.mean(DT_list_rmse))
        NLinear_nrmse.append(np.mean(LR_list_rmse))
        NDT_nrmse.append(np.mean(DT_list_nrmse))
        NLinear_nrmse.append(np.mean(LR_list_nrmse))
        row_contents = [str(normloop), str(Linear_rmse), str(DT_rmse), str(NLinear_rmse), str(NDT_rmse)]
        append_list_as_row(csvFileName, row_contents)
    else:
        print("Participant does not exist")
