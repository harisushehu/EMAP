# -*- coding: utf-8 -*-
"""
Created on Fri Aug 5 21:30:09 2022
@author: harisushehu
"""

import pandas as pd
import glob
import numpy as np
import pyswarms as ps
import os
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn import metrics
from sklearn.metrics import r2_score

def nrmse(rmse, y_test):
    nrmse = (rmse) / ((max(y_test) - min(y_test)))
    return nrmse[0]

# Save results
import csv
from csv import writer

# Append data in CSV function
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

csvFilename = '../results/results_LR_shift0.csv'

# Read in CSV file
if os.path.exists(csvFilename):
    print()
else:
    with open(csvFilename, 'w', newline='') as f:
        header = ['Fold', 'Before_RMSE', 'Before_NRMSE', 'After_RMSE', 'After_NRMSE'] 
        filewriter = csv.DictWriter(f, fieldnames=header)
        filewriter.writeheader()

print("Reading data...")

path = '' #path to dataset
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
    li.append(df)

dataset = pd.concat(li, axis=0, ignore_index=True)

# Replace NaN with 0
dataset = dataset.fillna(0)

print("Evaluating...")

X = dataset.iloc[:, df.columns != 'GSR_mean']
y = dataset.iloc[:, df.columns == 'GSR_mean'].values

print("X is:", X.shape)
print("y is:", y.shape)

print("Reading data for preprocessing...")
path = '' #path
all_files = sorted(glob.glob(path + "/*.csv"))

psysio = []
label = []

count = 0
numInit = 0

full_list = []
first_part = []
second_part = []
third_part = []
fourth_part = []
fifth_part = []

import random

participants_list = []

for k in range(1, 154):
    participants_list.append(k)

random.shuffle(participants_list)

for normloop in range(0, 153):  # 154
    participant = participants_list[normloop]
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

list_of_lists = [first_part, second_part, third_part, fourth_part, fifth_part]
for i in range(len(list_of_lists)):
    print("***********Splitting test and train...")
    train_lists = list_of_lists[0:i] + list_of_lists[i + 1:]
    test_list = list_of_lists[i]
    print("***********Splitting train and eval...")
    second_list_of_lists = train_lists
    for j in range(0, 1):
        if i < len(list_of_lists) - 1:
            train_lists = second_list_of_lists[0:j] + second_list_of_lists[j + 1:]
            val_list = second_list_of_lists[i]
        else:
            train_lists = second_list_of_lists[0:j] + second_list_of_lists[j + 1:]
            val_list = second_list_of_lists[j]

    import itertools
    train = list(itertools.chain(*train_lists))

    train_data = pd.concat(train, axis=0, ignore_index=True)

    # Replace NaN with 0
    train_data = train_data.fillna(0)

    train_X = train_data.iloc[:, df.columns != 'GSR_mean']
    train_y = train_data.iloc[:, df.columns == 'GSR_mean'].values

    train_X['GSR_mean'] = train_y
    X_train = train_data.iloc[:, df.columns != 'LABEL_SR_Arousal']
    y_train = train_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    # Scale train
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Normalize train X and y
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    test_data = pd.concat(test_list, axis=0, ignore_index=True)

    # Replace NaN with 0
    test_data = test_data.fillna(0)

    test_X = test_data.iloc[:, df.columns != 'GSR_mean']
    test_y = test_data.iloc[:, df.columns == 'GSR_mean'].values

    test_X['GSR_mean'] = test_y
    X_test = test_data.iloc[:, df.columns != 'LABEL_SR_Arousal']
    y_test = test_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    # Scale test
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Normalize test X and y
    X_test = scaler_X.fit_transform(X_test)
    y_test = scaler_y.fit_transform(y_test)

    val_data = pd.concat(val_list, axis=0, ignore_index=True)

    # Replace NaN with 0
    val_data = val_data.fillna(0)

    val_X = val_data.iloc[:, df.columns != 'GSR_mean']
    val_y = val_data.iloc[:, df.columns == 'GSR_mean'].values

    val_X['GSR_mean'] = val_y

    X_val = val_data.iloc[:, df.columns != 'LABEL_SR_Arousal']
    y_val = val_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    # Scale val
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Normalize test X and y
    X_val = scaler_X.fit_transform(X_val)
    y_val = scaler_y.fit_transform(y_val)

    from sklearn.linear_model import LinearRegression

    # ------------------------------------Before------------------------------------------>

    print("Before Feature selection.......")
    
    # Create an instance of the classifier
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    before_rmse = sqrt(metrics.mean_squared_error(y_test, y_pred))
    print("RMSE", before_rmse)
    before_nrmse = nrmse(before_rmse, y_test)
    print("NRMSE", before_nrmse)

    import time
    start_time = time.time()

    # Define objective function
    def f_per_particle(m, alpha):
        """Computes for the objective function per particle

        Inputs
        ------
        m : numpy.ndarray
            Binary mask that can be obtained from BinaryPSO, will
            be used to mask features.
        alpha: float (default is 0.5)
            Constant weight for trading-off classifier performance
            and number of features

        Returns
        -------
        numpy.ndarray
            Computed objective function
        """
        total_features = X.shape[1]

        X_subset = X_train[:, m > 0.5]
        reg = LinearRegression().fit(X_subset, y_train)

        X_eval_subset = X_val[:, m > 0.5]
        y_pred_eval = reg.predict(X_eval_subset)

        P = r2_score(y_val, y_pred_eval)

        j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

        return j

    def f(x, alpha=0.88):
        """Higher-level method to do classification in the
        whole swarm.

        Inputs
        ------
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search

        Returns
        -------
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        n_particles = x.shape[0]
        j = [f_per_particle(x[i], alpha) for i in range(n_particles)]

        return np.array(j)

    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 30, 'p': 2}

    # Call instance of PSO
    dimensions = X_train.shape[1]
    
    min_val = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    max_val = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    for i in range(0, 25):
        min_val = np.append(min_val, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        max_val = np.append(max_val, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    values_bound = (min_val, max_val)

    optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=dimensions, options=options, bounds=values_bound)

    # Perform optimization
    cost, pos = optimizer.optimize(f, iters=100)  # 1000

    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken ", time_taken)
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    X_selected_features = X_train[:, pos > 0.5]
    reg = LinearRegression().fit(X_selected_features, y_train)
    X_test = X_test[:, pos > 0.5]

    print("After Feature selection.......")

    print("Predicting test set...")
    y_pred_test = reg.predict(X_test)

    after_rmse = sqrt(metrics.mean_squared_error(y_test, y_pred_test))
    print("After RMSE is:", after_rmse)

    after_rmse = sqrt(metrics.mean_squared_error(y_test, y_pred_test))
    after_nrmse = nrmse(after_rmse, y_test)
    print("After NRMSE is:", after_nrmse)

    row_contents = [str(i), str(before_rmse), str(before_nrmse), str(after_rmse), str(after_nrmse)]
    append_list_as_row(csvFilename, row_contents)

    iterate = random.random()
    finalFeatures = "FS_LR0_pos" + str(iterate) + ".csv"

    if os.path.exists(finalFeatures):
        print()
    else:
        with open(finalFeatures, 'a+', newline='') as f:
            header = ['Iteration', 'pos']
            filewriter = csv.DictWriter(f, fieldnames=header)
            filewriter.writeheader()

    for i in range(0, len(pos)):
        row_contents = [str(i), str(pos[i])]
        append_list_as_row(finalFeatures, row_contents)

    count = 0
    for k in range(0, len(pos)):
        if pos[k] > 0.5:
            count = count + 1

    row_contents = [" ", " "]
    append_list_as_row(finalFeatures, row_contents)

    row_contents = ["Selected", str(count)]
    append_list_as_row(finalFeatures, row_contents)

    row_contents = [" ", " "]
    append_list_as_row(finalFeatures, row_contents)

    row_contents = ["RMSE Before", str(before_rmse)]
    append_list_as_row(finalFeatures, row_contents)

    row_contents = ["NRMSE Before", str(before_nrmse)]
    append_list_as_row(finalFeatures, row_contents)

    row_contents = ["RMSE After", str(after_rmse)]
    append_list_as_row(finalFeatures, row_contents)

    row_contents = ["NRMSE After", str(after_nrmse)]
    append_list_as_row(finalFeatures, row_contents)

    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_takes = str(hours) + ":" + str(minutes) + ":" + str(seconds)
    row_contents = ["Time Taken", time_takes]
    append_list_as_row(finalFeatures, row_contents)

    print("LR Shift 0...")
