# -*- coding: utf-8 -*-
"""
Created on Sat May 28 22:40:14 2022

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

def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

csvFileName = "../results/groupBased_categorization_overallRatings.csv"

if not os.path.exists(csvFileName):
    with open(csvFileName, 'a+', newline='') as f:
        header = ['Iteration', 'DT', 'RF']
        filewriter = csv.DictWriter(f, fieldnames=header)
        filewriter.writeheader()

print("Reading data...")

#path to your data
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

print("Reading data for preprocessing...")
#path to your data
path = ''
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

ZR_list = []
RP_list = []

ZR_list1 = []

import random

participants_list = []

for k in range(1, 154):
    participants_list.append(k)

random.shuffle(participants_list)

for normloop in range(0, 153):  # 153
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

            overall = filename.split("Features_P")
            participant_name = overall[1].split("-")
            participant_name = participant_name[0]

            overall = overall[1].split("T")
            overall = overall[1].split(".csv")
            overall = int(overall[0])

            valencePath = "./SessionInfo/P" + participant_name + "-Session.csv"
            Valence = pd.read_csv(valencePath, encoding='ISO-8859-1', header=0)

            for k in range(0, len(Valence)):
                if Valence['trialNumber'][k] == overall:
                    ind = k

            overallRatings = Valence['respArousal'][ind]

            reader = reader.fillna(0)

            if reader['LABEL_SR_Arousal'].any() == True:
                for i in range(0, len(reader)):
                    reader['LABEL_SR_Arousal'][i] = overallRatings

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

    import itertools
    train = list(itertools.chain(*train_lists))

    train_data = pd.concat(train, axis=0, ignore_index=True)
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

    count0 = 0
    count1 = 0
    for i in range(0, len(y_train)):
        if y_train[i] == 0:
            count0 = count0 + 1
        else:
            count1 = count1 + 1

    print("There is a total of", str(count0) + " 0's and", str(count1) + " 1's in training data...")

    count0 = 0
    count1 = 0
    for i in range(0, len(y_test)):
        if y_test[i] == 0:
            count0 = count0 + 1
        else:
            count1 = count1 + 1

    print("There is a total of", str(count0) + " 0's and", str(count1) + " 1's in test data....")

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

    print("[INFO] using '{}' model".format(args["model"]))
    model = models[args["model"]]
    model.fit(X_train, y_train)

    target_names = ['0=low', '1=high']

    print("[INFO] evaluating...")
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, target_names=target_names))

    print(confusion_matrix(y_test, predictions))

    y_pred = model.predict(X_test)

    res_DT = accuracy_score(y_test, y_pred, normalize=True)

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
    res_RF = accuracy_score(y_test, y_pred, normalize=True)

    print("DT avg:", res_DT)
    print("RF avg:", res_RF)

    row_contents = [str(normloop), str(res_DT), str(res_RF)]
    append_list_as_row(csvFileName, row_contents)
