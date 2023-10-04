# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:27:18 2022
@author: harisushehu
"""

import pandas as pd
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import statistics
import csv
from csv import writer
from random import randrange, shuffle

# Function to append a list as a row to a CSV file
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

# Define the output CSV file
csvFileName = "../results/participantBased_categorization_random.csv"

# Read in CSV file or create a new one with a header if it doesn't exist
if os.path.exists(csvFileName):
    print()
else:
    with open(csvFileName, 'w', newline='') as f:
        header = ['Participant', 'ZeroRule', 'RandomPrediction'] 
        filewriter = csv.DictWriter(f, fieldnames = header)
        filewriter.writeheader()

# Zero Rule Algorithm for classification
def zero_rule_algorithm_classification(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for i in range(len(test))]
    return predicted

# Generate random predictions
def random_algorithm(train, test):
    output_values = [row[-1] for row in train]
    unique = list(set(output_values))
    predicted = list()
    for row in test:
        index = randrange(len(unique))
        predicted.append(unique[index])
    return predicted

print("Reading data...")

#path to your data
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

X = dataset.iloc[:, df.columns != 'LABEL_SR_Arousal']
y = dataset.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

print("X is:", X.shape)
print("y is:", y.shape)

print("Reading data for preprocessing...")
#path to your data
path = ''
all_files = sorted(glob.glob(path + "/*.csv"))

full_list = []

for normloop in range(0, 154):

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

        # Drop rows with NaN
        test_data = test_data.dropna()

        import numpy as np
        test_X = test_data.iloc[:, test_data.columns != 'LABEL_SR_Arousal']
        test_X = np.array(test_X)
        test_y = test_data.iloc[:, test_data.columns == 'LABEL_SR_Arousal'].values

        # Scale test data
        scaler_X2 = StandardScaler()
        scaler_y2 = StandardScaler()

        test_X = scaler_X2.fit_transform(test_X)
        test_y = scaler_y2.fit_transform(test_y)

        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import classification_report

        ZR_list = []
        RP_list = []

        # Scale y_test values
        for i in range(0, len(test_y)):
            if test_y[i] <= 0.5:
                test_y[i] = 0
            else:
                test_y[i] = 1

        # Prepare cross-validation
        kfold = StratifiedKFold(3)

        # Enumerate splits
        for train_index, test_index in kfold.split(test_X, test_y):

            X_train, X_test = test_X[train_index], test_X[test_index]
            y_train, y_test = test_y[train_index], test_y[test_index]

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

            count0 = sum(y_train == 0)
            count1 = sum(y_train == 1)

            print("There are a total of", count0, "0's and", count1, "1's in the training data...")

            count0 = sum(y_test == 0)
            count1 = sum(y_test == 1)

            print("There are a total of", count0, "0's and", count1, "1's in the test data....")

            print("[INFO] using '{}' model".format(args["model"]))
            model = models[args["model"]]
            model.fit(X_train, y_train)

            target_names = ['0=low', '1=high']

            print("[INFO] evaluating...")
            y_pred = zero_rule_algorithm_classification(y_train, y_test)
            print(classification_report(y_test, y_pred, target_names=target_names))

            from sklearn.metrics import confusion_matrix
            print(confusion_matrix(y_test, y_pred))

            from sklearn.metrics import accuracy_score
            print("Accuracy with Zero Rule Classifier", accuracy_score(y_test, y_pred, normalize=True))
            ZR_list.append(accuracy_score(y_test, y_pred, normalize=True))

            ap = argparse.ArgumentParser()
            ap.add_argument("-d", "--dataset", type=str, default="aligned",
                            help="path to directory containing the 'emotions' dataset")
            ap.add_argument("-m", "--model", type=str, default="decision_tree",
                            help="type of python machine learning model to use")
            args = vars(ap.parse_args())

            models = {
                "knn": KNeighborsClassifier(n_neighbors=1),
                "naive_bayes": GaussianNB(),
                "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
                "svm": SVC(kernel="linear", degree=8),
                "svm1": SVC(kernel='linear', probability=True, tol=1e-3),
                "decision_tree": DecisionTreeClassifier(),
                "random_forest": RandomForestClassifier(n_estimators=2),
                "mlp": MLPClassifier()
            }

            print("[INFO] using '{}' model".format(args["model"]))
            model = models[args["model"]]
            model.fit(X_train, y_train)

            target_names = ['0=low', '1=high']

            print("[INFO] evaluating...")
            y_pred = random_algorithm(y_train, y_test)
            print(classification_report(y_test, y_pred, target_names=target_names))

            from sklearn.metrics import confusion_matrix
            print(confusion_matrix(y_test, y_pred))

            from sklearn.metrics import accuracy_score
            print("Accuracy with Random Classifier", accuracy_score(y_test, y_pred, normalize=True))

            RP_list.append(accuracy_score(y_test, y_pred, normalize=True))

        # Average of fold results
        ZR_avg = statistics.mean(ZR_list)
        RP_avg = statistics.mean(RP_list)

        print("****************************************************")
        print("Results for fold no. " + str(normloop))

        print("Zero Rule avg:", ZR_avg)
        print("Random Prediction avg:", RP_avg)

        # Append and save results
        row_contents = [str(normloop), str(ZR_avg), str(RP_avg)]
        append_list_as_row(csvFileName, row_contents)

    else:
        print("Participant does not exist")
