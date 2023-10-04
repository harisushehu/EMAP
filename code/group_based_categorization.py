# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:22:40 2021

@author: harisushehu
"""


import pandas as pd
import glob
import os
import csv
from csv import writer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import random

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from the csv module
        csv_writer = writer(write_obj)
        # Add contents of the list as the last row in the CSV file
        csv_writer.writerow(list_of_elem)

# CSV file to store results
csvFileName = "../results/groupBased_categorization_balanced.csv"

# Check if the CSV file exists, if not create it with headers
if not os.path.exists(csvFileName):
    with open(csvFileName, 'a+', newline='') as f:
        header = ['DT', 'RF']
        filewriter = csv.DictWriter(f, fieldnames=header)
        filewriter.writeheader()

# Function to balance the dataset
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
    for i in range(0, len(test_y)-1):
        if countRemoved == difference:
            break
        
        if ((i < len(test_y)) and (test_y[i] == identifier) and (countRemoved < difference)):
            test_y = np.delete(test_y, obj=i, axis=0)
            test_X = np.delete(test_X, obj=i, axis=0)
            countRemoved += 1
            
    return test_y, test_X             

print("Reading data...")
# Specify the path to your data files
path = ''

# Use glob to get a list of all CSV files in the specified directory
all_files = glob.glob(path + "/*.csv")

li = []

# Read each CSV file and append it to a list
for filename in all_files:
    df = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
    li.append(df)

# Concatenate all the dataframes into one
dataset = pd.concat(li, axis=0, ignore_index=True)

# Replace NaN with 0
dataset = dataset.fillna(0)

print("Evaluating...")

print("Reading data for preprocessing...")

# Use glob to get a list of all CSV files in the specified directory
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

DT_list = []
RF_list = []

participants_list = []

# Generate a shuffled list of participants
for k in range(1, 154):
    participants_list.append(k)

random.shuffle(participants_list)

for normloop in range(0, 153):
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
    train_lists = list_of_lists[0:i] + list_of_lists[i+1:]
    test_list = list_of_lists[i]
    import itertools
    train = list(itertools.chain(*train_lists))
    
    train_data = pd.concat(train, axis=0, ignore_index=True)
            
    # Replace NaN with 0
    train_data = train_data.fillna(0)
    
    X_train = train_data.iloc[:, df.columns != 'LABEL_SR_Arousal']
    y_train = train_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    test_data = pd.concat(test_list, axis=0, ignore_index=True)
            
    test_data = test_data.fillna(0)
    
    X_test = test_data.iloc[:, df.columns != 'LABEL_SR_Arousal']
    y_test = test_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_test = scaler_X.fit_transform(X_test)
    y_test = scaler_y.fit_transform(y_test)
    
    for i in range(0, len(y_train)):
        if y_train[i] <= 0.5:     
            y_train[i] = 0    
        else: 
            y_train[i] = 1
    
    for i in range(0, len(y_test)):
        if y_test[i] <= 0.5:     
            y_test[i] = 0    
        else: 
            y_test[i] = 1
            
    y_test, X_test = ratioedData(y_test, X_test)
    y_train, X_train = ratioedData(y_train, X_train)
            
    count0 = 0
    count1 = 1
    for i in range(0, len(y_train)):
        if y_train[i] == 0:     
            count0 = count0 + 1
        else: 
            count1 = count1 + 1
            
    print("There is a total of ", str(count0) + " 0's and ", str(count1) + " 1's in the training data...")
    
    count0 = 0
    count1 = 1
    for i in range(0, len(y_test)):
        if y_test[i] == 0:     
            count0 = count0 + 1
        else: 
            count1 = count1 + 1
            
    print("There is a total of ", str(count0) + " 0's and ", str(count1) + " 1's in the test data....")

    # ---------------------------------***Classification***---------------------------------------------->
    
    models = {
        "knn": KNeighborsClassifier(n_neighbors=1),
        "naive_bayes": GaussianNB(),
        "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
        "svm": SVC(kernel="linear", degree=8),
        "svm1": SVC(kernel='linear', probability=True, tol=1e-3),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(n_estimators=120),
        "mlp": MLPClassifier()
    }
    
    for model_name, model in models.items():
        print("[INFO] using '{}' model".format(model_name))
        model.fit(X_train, y_train)
    
        target_names = ['0=low', '1=high']
    
        print("[INFO] evaluating...")
        predictions = model.predict(X_test)
        print(classification_report(y_test, predictions, target_names=target_names))
    
        print(confusion_matrix(y_test, predictions))
    
        y_pred = model.predict(X_test)
    
        print("Accuracy with {}: {:.2f}".format(model_name, accuracy_score(y_test, y_pred, normalize=True)))
    
        if model_name == "decision_tree":
            DT_list.append(accuracy_score(y_test, y_pred, normalize=True))
        elif model_name == "random_forest":
            RF_list.append(accuracy_score(y_test, y_pred, normalize=True))

# Saving results for DT & RF in CSV
for j in range(0, len(DT_list)):
    row_contents = [str(DT_list[j]), str(RF_list[j])]
    append_list_as_row(csvFileName, row_contents)
