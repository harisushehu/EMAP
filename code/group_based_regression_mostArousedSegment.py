# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:36:19 2022
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


# Function to append a list as a row to a CSV file
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as the last row in the CSV file
        csv_writer.writerow(list_of_elem)


# Function to calculate normalized root mean squared error (NRMSE)
def nrmse(rmse, y_test):
    nrmse = (rmse) / ((max(y_test) - min(y_test)))
    return nrmse[0]


# Define the output CSV file
csvFileName = "../results/groupBased_regression_mostArousedSegment.csv"

# Read in CSV file or create a new one with a header if it doesn't exist
if os.path.exists(csvFileName):
    print()
else:
    with open(csvFileName, 'a+', newline='') as f:
        header = ['Iteration', 'LR-RMSE', 'DT-RMSE', 'LR-NRMSE', 'DT-NRMSE']
        filewriter = csv.DictWriter(f, fieldnames=header)
        filewriter.writeheader()

print("Reading data...")

# Define the path to the data files
path = ''
all_files = glob.glob(path + "/*.csv")

li = []

# Read and concatenate all data files
for filename in all_files:
    df = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
    li.append(df)

dataset = pd.concat(li, axis=0, ignore_index=True)

# Replace NaN with 0
dataset = dataset.fillna(0)

print("Evaluating...")

print("Reading data for preprocessing...")

# Define the path to data files again
all_files = sorted(glob.glob(path + "/*.csv"))

count = 0
full_list = []
first_part = []
second_part = []
third_part = []
fourth_part = []
fifth_part = []

import random

participants_list = []

# Create a list of participant numbers and shuffle it
for k in range(1, 154):
    participants_list.append(k)

random.shuffle(participants_list)

# Initialize lists to store RMSE and NRMSE results
DT_NRMSE_all = []
LR_NRMSE_all = []

# Iterate through participants
for normloop in range(0, 153):

    participant = participants_list[normloop]

    flag = False
    fileReader = []

    increament = 0

    # Iterate through data files
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
                # Most emotionally aroused segment
                maximum = max(reader['LABEL_SR_Arousal'])
                for i in range(0, len(reader)):
                    reader['LABEL_SR_Arousal'][i] = maximum

            fileReader.append(reader)
            flag = True

    if flag == True:
        full_list.extend(fileReader)
        count = count + 1

    if (count / 29) == 1:
        first_part.extend(full_list)
        full_list = []
    elif (count / 29) == 2:
        second_part.extend(full_list)
        full_list = []
    elif (count / 29) == 3:
        third_part.extend(full_list)
        full_list = []
    elif (count / 29) == 4:
        fourth_part.extend(full_list)
        full_list = []
    elif (count / 29) == 5:
        fifth_part.extend(full_list)
        full_list = []

# Combine the data lists
list_of_lists = [first_part, second_part, third_part, fourth_part, fifth_part]

# Iterate through the data lists for cross-validation
for i in range(len(list_of_lists)):

    print("***********Splitting test and train...")
    train_lists = list_of_lists[0:i] + list_of_lists[i + 1:]
    test_list = list_of_lists[i]

    # Combine the train data
    import itertools
    train = list(itertools.chain(*train_lists))
    train_data = pd.concat(train, axis=0, ignore_index=True)

    # Replace NaN with 0
    train_data = train_data.fillna(0)
    X_train = train_data.iloc[:, df.columns != 'LABEL_SR_Arousal']
    y_train = train_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    # Scale train data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    test_data = pd.concat(test_list, axis=0, ignore_index=True)

    # Replace NaN with 0
    test_data = test_data.fillna(0)
    X_test = test_data.iloc[:, df.columns != 'LABEL_SR_Arousal']
    y_test = test_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    # Scale test data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_test = scaler_X.fit_transform(X_test)
    y_test = scaler_y.fit_transform(y_test)

    #---------------------------------***Regression***----------------------------------------------

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

    # Append results to the CSV file
    row_contents = [str(i), str(Linear_rmse), str(DT_rmse), str(NLinear_rmse), str(NDT_rmse)]
    append_list_as_row(csvFileName, row_contents)
