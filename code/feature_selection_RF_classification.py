# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 21:22:32 2022
@author: harisushehu
"""

import pandas as pd
import glob
import numpy as np
import pyswarms as ps
import os
from sklearn.preprocessing import StandardScaler

# Scoring imports
from sklearn.metrics import accuracy_score

# Save results
import csv
from csv import writer

# Append data in CSV function
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from the csv module
        csv_writer = writer(write_obj)
        # Add contents of list as the last row in the csv file
        csv_writer.writerow(list_of_elem)

csvFilename = '../results/results_RF_classiciation.csv'

# Read in CSV file
if os.path.exists(csvFilename):
    print()
else:
    with open(csvFilename, 'w', newline='') as f:
        header = ['Fold', 'Before_Acc', 'After_Acc']
        filewriter = csv.DictWriter(f, fieldnames=header)
        filewriter.writeheader()

print("Reading data...")

# path = '../Clean_Data' # use your path # use your path
path = ''
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

# Randomized data
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

    X_train = train_data.iloc[:, df.columns != 'LABEL_SR_Arousal']
    y_train = train_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    # Scale y_train values
    for i in range(0, len(y_train)):

        if y_train[i] <= 0.5:
            y_train[i] = 0
        else:
            y_train[i] = 1

    # Scale train
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Normalize train X
    X_train = scaler_X.fit_transform(X_train)

    test_data = pd.concat(test_list, axis=0, ignore_index=True)

    # Replace NaN with 0
    test_data = test_data.fillna(0)

    X_test = test_data.iloc[:, df.columns != 'LABEL_SR_Arousal']
    y_test = test_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    # Scale y_test values
    for i in range(0, len(y_test)):

        if y_test[i] <= 0.5:
            y_test[i] = 0
        else:
            y_test[i] = 1

    # Scale test
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Normalize test X
    X_test = scaler_X.fit_transform(X_test)

    val_data = pd.concat(val_list, axis=0, ignore_index=True)

    # Replace NaN with 0
    val_data = val_data.fillna(0)

    X_val = val_data.iloc[:, df.columns != 'LABEL_SR_Arousal']
    y_val = val_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    # Scale y_test values
    for i in range(0, len(y_val)):

        if y_val[i] <= 0.5:
            y_val[i] = 0
        else:
            y_val[i] = 1

    # Scale val
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Normalize test X and y
    X_val = scaler_X.fit_transform(X_val)

    # ------------------------------------Before------------------------------------------>
    from sklearn.ensemble import RandomForestClassifier

    print("Before Feature selection.......")
    
    # Create an instance of the classifier
    y_train = y_train.ravel()
    clf = RandomForestClassifier(n_estimators=2).fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    before_acc = accuracy_score(y_test, y_pred)
    print("Accuracy : ", before_acc)

    import time
    start_time = time.time()

    # Define objective function
    def f_per_particle(m, alpha):
        """Computes the objective function per particle

        Inputs
        ------
        m : numpy.ndarray
            Binary mask that can be obtained from BinaryPSO, will
            be used to mask features.
        alpha: float (default is 0.5)
            Constant weight for trading-off classifier performance
            and the number of features

        Returns
        -------
        numpy.ndarray
            Computed objective function
        """
        total_features = X.shape[1]
        X_subset = X_train[:, m > 0.5]
        clf = RandomForestClassifier(n_estimators=2).fit(X_subset, y_train)

        X_eval_subset = X_val[:, m > 0.5]
        y_pred_eval = clf.predict(X_eval_subset)

        P = accuracy_score(y_val, y_pred_eval)

        j = (alpha * (1.0 - P)
             + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

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

    # Call an instance of PSO
    dimensions = X_train.shape[1]

    min_val = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    max_val = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # Make 260 bounds
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

    # Get the selected features from the final positions
    X_selected_features = X_train[:, pos > 0.5]  # subset

    clf_ = RandomForestClassifier(n_estimators=2).fit(X_selected_features, y_train)
    X_test = X_test[:, pos > 0.5]

    # Compute performance
    print("After Feature selection.......")
    
    # Predict test results
    print("Predicting test set...")
    y_pred_test = clf_.predict(X_test)

    after_acc = accuracy_score(y_test, y_pred_test)
    print("After Accuracy is: ", after_acc)

    # Save results
    row_contents = [str(i), str(before_acc), str(after_acc)]
    
    # Append a list as a new line to an old CSV file
    append_list_as_row(csvFilename, row_contents)

    iterate = random.random()
    finalFeatures = "../results/RF_classification/FS_RF_pos" + str(iterate) + ".csv"

    # Read in CSV file
    if os.path.exists(finalFeatures):
        print()
    else:
        with open(finalFeatures, 'a+', newline='') as f:
            header = ['Iteration', 'pos']
            filewriter = csv.DictWriter(f, fieldnames=header)
            filewriter.writeheader()

    for i in range(0, len(pos)):
        row_contents = [str(i), str(pos[i])]
        # Append a list as a new line to an old CSV file
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

    # Accuracy Before FS
    row_contents = ["Accuracy Before", str(before_acc)]
    append_list_as_row(finalFeatures, row_contents)

    # Accuracy After FS
    row_contents = ["Accuracy After", str(after_acc)]
    append_list_as_row(finalFeatures, row_contents)

    # Time
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_takes = str(hours) + ":" + str(minutes) + ":" + str(seconds)
    row_contents = ["Time Taken", time_takes]
    append_list_as_row(finalFeatures, row_contents)
