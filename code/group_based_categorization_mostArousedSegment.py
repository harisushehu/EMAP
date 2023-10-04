# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:52:46 2022
@author: harisushehu
"""

import pandas as pd
import glob
import os
import csv
from csv import writer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import argparse
import random

# Function to append a list as a row to a CSV file
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

# Define the CSV file to store results
csvFileName = "../results/groupBased_categorization_mostArousedSegment.csv"
    
# Check if the CSV file exists, if not create it with headers
if not os.path.exists(csvFileName):
    with open(csvFileName, 'a+', newline='') as f:
        header = ['Iteration', 'DT', 'RF'] 
        filewriter = csv.DictWriter(f, fieldnames=header)
        filewriter.writeheader()

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

count = 0

full_list = []
first_part = []
second_part = []
third_part = []
fourth_part = []
fifth_part = []

ZR_list = []
RP_list = []

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
            lines= len(reader)
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
        
    if (count/29) == 1:
        first_part.extend(full_list)
        full_list = []
    elif (count/29) == 2:
        second_part.extend(full_list)
        full_list = []
    elif (count/29) == 3:
        third_part.extend(full_list)
        full_list = []
    elif (count/29) == 4:
        fourth_part.extend(full_list)
        full_list = []
    elif (count/29) == 5:
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
    
    # Scale train                
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
        
    # Normalize train X and y
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    test_data = pd.concat(test_list, axis=0, ignore_index=True)
            
    # Replace NaN with 0
    test_data = test_data.fillna(0)
    
    X_test = test_data.iloc[:, df.columns != 'LABEL_SR_Arousal']
    y_test = test_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values
    
    # Scale test                
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Normalize test X and y
    X_test = scaler_X.fit_transform(X_test)
    y_test = scaler_y.fit_transform(y_test)

    # Scale y_train values
    for i in range(0, len(y_train)):
        if y_train[i] <= 0.5:     
            y_train[i] = 0    
        else: 
            y_train[i] = 1
            
    # Scale y_test values
    for i in range(0, len(y_test)):
        if y_test[i] <= 0.5:     
            y_test[i] = 0    
        else: 
            y_test[i] = 1

    count0 = y_train.tolist().count(0)
    count1 = y_train.tolist().count(1)
    print("There is a total of {} 0's and {} 1's in training data...".format(count0, count1))
    
    count0 = y_test.tolist().count(0)
    count1 = y_test.tolist().count(1)
    print("There is a total of {} 0's and {} 1's in test data....".format(count0, count1))

    #---------------------------------***Classification***---------------------------------------------->
    
    target_names = ['0=low', '1=high']
    
    # Evaluate Decision Tree (DT) Classifier
    print("[INFO] evaluating Decision Tree (DT) Classifier...")
    
    # Define and train the DT model
    model_DT = DecisionTreeClassifier()
    model_DT.fit(X_train, y_train)
    
    # Make predictions on test data and show a classification report
    y_pred_DT = model_DT.predict(X_test)
    print(classification_report(y_test, y_pred_DT, target_names=target_names))
    
    # Confusion matrix for DT Classifier
    print(confusion_matrix(y_test, y_pred_DT))
    
    # Calculate accuracy and append results
    accuracy_DT = accuracy_score(y_test, y_pred_DT, normalize=True)
    print("Accuracy with Decision Tree (DT) Classifier: {:.2f}".format(accuracy_DT))
    
    # Evaluate Random Forest (RF) Classifier
    print("[INFO] evaluating Random Forest (RF) Classifier...")
    
    # Define and train the RF model
    model_RF = RandomForestClassifier(n_estimators=2)
    model_RF.fit(X_train, y_train)
    
    # Make predictions on test data and show a classification report
    y_pred_RF = model_RF.predict(X_test)
    print(classification_report(y_test, y_pred_RF, target_names=target_names))
    
    # Confusion matrix for RF Classifier
    print(confusion_matrix(y_test, y_pred_RF))
    
    # Calculate accuracy and append results
    accuracy_RF = accuracy_score(y_test, y_pred_RF, normalize=True)
    print("Accuracy with Random Forest (RF) Classifier: {:.2f}".format(accuracy_RF))
    
    print("DT avg: ", accuracy_DT)
    print("RF avg: ", accuracy_RF)
    
    # Append and save results to CSV
    row_contents = [str(normloop), str(accuracy_DT), str(accuracy_RF)]
    append_list_as_row(csvFileName, row_contents)
