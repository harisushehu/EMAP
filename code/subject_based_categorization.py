# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 21:06:04 2021

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
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import argparse
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

csvFileName = "../results/Full_data/EMAP_ParticipantBased_categorization_TML_1_1.csv"

if os.path.exists(csvFileName):
    print()
else:
    with open(csvFileName, 'w', newline='') as f:
        header = ['Participant', 'DT', 'RF']
        filewriter = csv.DictWriter(f, fieldnames=header)
        filewriter.writeheader()

def ratioedData(test_y, test_X):
    countOnes = np.count_nonzero(test_y == 1)
    countZeros = np.count_nonzero(test_y == 0)

    if countOnes > countZeros:
        identifier = 1
    else:
        identifier = 0

    difference = abs(countZeros - countOnes)
    countRemoved = 0

    for i in range(0, len(test_y) - 1):
        if countRemoved == difference:
            break

        if ((i < len(test_y)) and (test_y[i] == identifier) and (countRemoved < difference)):
            test_y = np.delete(test_y, obj=i, axis=0)
            test_X = np.delete(test_X, obj=i, axis=0)
            countRemoved += 1

    return test_y, test_X

print("Reading data...")

path = '//smb/grid-solar/sgeusers/harisushehu/EMAP/Features_TAC'
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
    li.append(df)

dataset = pd.concat(li, axis=0, ignore_index=True)
dataset = dataset.fillna(0)

print("Evaluating...")

X = dataset.iloc[:, df.columns != 'LABEL_SR_Arousal']
y = dataset.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

print("X is :", X.shape)
print("y is :", y.shape)

print("Reading data for preprocessing...")
path = '//smb/grid-solar/sgeusers/harisushehu/EMAP/Features_TAC'
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
        test_data = test_data.dropna()
        test_X = test_data.iloc[:, test_data.columns != 'LABEL_SR_Arousal']
        test_X = np.array(test_X)
        test_y = test_data.iloc[:, test_data.columns == 'LABEL_SR_Arousal'].values

        scaler_X2 = StandardScaler()
        scaler_y2 = StandardScaler()

        test_X = scaler_X2.fit_transform(test_X)
        test_y = scaler_y2.fit_transform(test_y)

        res_DT = []
        res_RF = []

        for i in range(0, len(test_y)):
            if test_y[i] <= 0.5:
                test_y[i] = 0
            else:
                test_y[i] = 1

        test_y, test_X = ratioedData(test_y, test_X)

        pred_list = []
        true_list = []

        kfold = StratifiedKFold(3)
        for train_index, test_index in kfold.split(test_X, test_y):
            X_train, X_test = test_X[train_index], test_X[test_index]
            y_train, y_test = test_y[train_index], test_y[test_index]

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
                "random_forest": RandomForestClassifier(n_estimators=120),
                "mlp": MLPClassifier()
            }

            count0 = 0
            count1 = 1
            for i in range(0, len(y_train)):
                if y_train[i] == 0:
                    count0 = count0 + 1
                else:
                    count1 = count1 + 1

            print("There is a total of ", str(count0) + " 0's and ", str(count1) + " 1's in training data...")

            count0 = 0
            count1 = 1
            for i in range(0, len(y_test)):
                if y_test[i] == 0:
                    count0 = count0 + 1
                else:
                    count1 = count1 + 1

            print("There is a total of ", str(count0) + " 0's and ", str(count1) + " 1's in test data....")

            print("[INFO] using '{}' model".format(args["model"]))
            model = models[args["model"]]
            model.fit(X_train, y_train)

            target_names = ['0=low', '1=high']

            print("[INFO] evaluating...")
            predictions = model.predict(X_test)
            print(classification_report(y_test, predictions, target_names=target_names))

            print(confusion_matrix(y_test, predictions))

            y_pred = model.predict(X_test)

            res_DT.append(accuracy_score(y_test, y_pred, normalize=True))

            ap = argparse.ArgumentParser()
            ap.add_argument("-d", "--dataset", type=str, default="aligned",
                            help="path to directory containing the 'emotions' dataset")
            ap.add_argument("-m", "--model", type=str, default="random_forest",
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
            predictions = model.predict(X_test)
            print(classification_report(y_test, predictions, target_names=target_names))

            print(confusion_matrix(y_test, predictions))

            y_pred = model.predict(X_test)

            for x in range(0, len(y_pred)):
                pred_list.append(y_pred[x])
                true_list.append(y_test[x])

            res_RF.append(accuracy_score(y_test, y_pred, normalize=True))

        DT_avg = statistics.mean(res_DT)
        RF_avg = statistics.mean(res_RF)

        print("****************************************************")
        print("Results for fold no. " + str(normloop))

        print("DT avg: ", DT_avg)
        print("RF avg: ", RF_avg)

        print("Overall confusion matrix is...")
        print(confusion_matrix(true_list, pred_list))

        row_contents = [str(normloop), str(DT_avg), str(RF_avg)]
        append_list_as_row(csvFileName, row_contents)
    else:
        print("Participant does not exist")
