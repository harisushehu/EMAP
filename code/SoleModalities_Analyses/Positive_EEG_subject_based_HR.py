#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:02:05 2024

@author: harisushehu
"""

import pandas as pd
import glob
import os
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression
from numpy import sqrt
from sklearn.metrics import confusion_matrix
from csv import writer

def nrmse(rmse, y_test): 
    nrmse = rmse / (max(y_test) - min(y_test))
    return nrmse[0]

# Append data to CSV
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

# Define paths and filenames
csvFileName = "../results/Positive_SoleModalities_SubjectBased_EEGRegression_HR.csv"
data_path = '/Users/harisushehu/Desktop/Features' #use your path

# Check if CSV file exists and create it if not
if not os.path.exists(csvFileName):
    with open(csvFileName, 'w', newline='') as f:
        header = ['Iteration', 'LR_RMSE', 'LR_NRMSE', 'DT_RMSE', 'DT_NRMSE'] 
        filewriter = writer(f, fieldnames=header)
        filewriter.writeheader()

print("Reading data...")

# Load data from multiple CSV files
all_files = glob.glob(os.path.join(data_path, "*.csv"))
li = []

for filename in all_files:
    df = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
    li.append(df)

dataset = pd.concat(li, axis=0, ignore_index=True)
dataset = dataset.fillna(0)

print("Evaluating...")

'''
# Drop columns 'GSR', 'IRPleth', 'Respir', and 'Arousal'
columns_to_drop = ['GSR_mean', 'IRPleth_mean', 'Respir_mean', 'LABEL_SR_Arousal']
dataset = dataset.drop(columns_to_drop, axis=1)
'''

# Extract features and labels
X = dataset.iloc[:, dataset.columns != 'heartrate_mean']
y = dataset.iloc[:, dataset.columns == 'heartrate_mean'].values

print("X shape:", X.shape)
print("y shape:", y.shape)

# Shuffle participants list
participants_list = list(range(1, 154))
random.shuffle(participants_list)

# Regression EEG
for normloop in range(0, 153):
    participant = participants_list[normloop]
    flag = False
    fileReader = []
    increament = 0
    
    for filename in all_files:
        partNo = str(participant).zfill(3)
        if partNo in filename:
            reader = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
            lines = len(reader)
            increament += lines
            
            session = filename.split("Features_P")
            participant_name = session[1].split("-")
            participant_name = participant_name[0]
            session = session[1].split("T")
            session = session[1].split(".csv")
            session = int(session[0])
            valencePath = "../SessionInfo/P"  + participant_name + "-Session.csv"
            Valence = pd.read_csv(valencePath, encoding='ISO-8859-1', header=0)
            
            for k in range(0, len(Valence)):
                if Valence['movieValence'][k] == 'Positive':
                    fileReader.append(reader)
                    
            flag = True
    
    if flag:
        print("Evaluating for participant " + str(normloop) + "...")
        full_list = fileReader
        test_reader = full_list
        full_list = []
        test_data = pd.concat(test_reader, axis=0, ignore_index=True)
        test_data = test_data.dropna()
        test_X = test_data.iloc[:, :256]
        test_X = np.array(test_X)
        test_y = test_data.iloc[:, test_data.columns == 'heartrate_mean'].values
        scaler_X2 = StandardScaler()
        scaler_y2 = StandardScaler()
        test_X = scaler_X2.fit_transform(test_X)
        test_y = scaler_y2.fit_transform(test_y)
        
        LR_list = []
        DT_list = []
        NLR_list = []
        NDT_list = []
        
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
                
            DT_list.append(DT_rmse)
            LR_list.append(Linear_rmse)
            NDT_list.append(NDT_rmse)
            NLR_list.append(NLinear_rmse)
        
        row_contents = [str(normloop), str(np.mean(LR_list)), str(np.mean(NLR_list)), str(np.mean(DT_list)), str(np.mean(NDT_list))]
        append_list_as_row(csvFileName, row_contents)
    else:
        print("Participant does not exist")