# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:15:14 2022

@author: harisushehu
"""

import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import argparse
from sklearn.metrics import classification_report
import os
import csv
from csv import writer
from random import randrange

# Function to append a list as a row to a CSV file
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

# Function to calculate normalized root mean square error
def nrmse(rmse, y_test): 
    nrmse = (rmse) / ((max(y_test) - min(y_test)))
    return nrmse[0]

# Define the CSV file name for storing results
csvFileName = "../results/SoleModalities_GroupBased_EEGRegression.csv"

# Check if the CSV file exists, if not, create it with header
if not os.path.exists(csvFileName):
    with open(csvFileName, 'w', newline='') as f:
        header = ['Iteration', 'LR_RMSE', 'LR_NRMSE', 'DT_RMSE', 'DT_NRMSE'] 
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

participants_list = []

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

# Regression EEG
list_of_lists = [first_part, second_part, third_part, fourth_part, fifth_part]
for i in range(len(list_of_lists)):
    train_lists = list_of_lists[0:i] + list_of_lists[i+1:]
    test_list = list_of_lists[i]

    import itertools
    train = list(itertools.chain(*train_lists))

    train_data = pd.concat(train, axis=0, ignore_index=True)
    train_data = train_data.fillna(0)

    X_train = train_data.iloc[:, :256]
    y_train = train_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    test_data = pd.concat(test_list, axis=0, ignore_index=True)
    test_data = test_data.fillna(0)

    X_test = test_data.iloc[:, :256]
    y_test = test_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_test = scaler_X.fit_transform(X_test)
    y_test = scaler_y.fit_transform(y_test)

    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from numpy import sqrt

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

    # Save results
    row_contents = [str(i), str(Linear_rmse), str(NLinear_rmse), str(DT_rmse), str(NDT_rmse)]
    append_list_as_row(csvFileName, row_contents)



# Classification EEG
csvFileName = "../results/SoleModalities_GroupBased_EEGClassification.csv"

if not os.path.exists(csvFileName):
    with open(csvFileName, 'w', newline='') as f:
        header = ['Iteration', 'DT', 'RF']
        filewriter = csv.DictWriter(f, fieldnames=header)
        filewriter.writeheader()

list_of_lists = [first_part, second_part, third_part, fourth_part, fifth_part]
for i in range(len(list_of_lists)):
    train_lists = list_of_lists[0:i] + list_of_lists[i+1:]
    test_list = list_of_lists[i]

    import itertools
    train = list(itertools.chain(*train_lists))

    train_data = pd.concat(train, axis=0, ignore_index=True)
    train_data = train_data.fillna(0)

    X_train = train_data.iloc[:, :256]
    y_train = train_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    test_data = pd.concat(test_list, axis=0, ignore_index=True)
    test_data = test_data.fillna(0)

    X_test = test_data.iloc[:, :256]
    y_test = test_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_test = scaler_X.fit_transform(X_test)
    y_test = scaler_y.fit_transform(y_test)

    for j in range(0, len(y_train)):
        if y_train[j] <= 0.5:
            y_train[j] = 0
        else:
            y_train[j] = 1

    for j in range(0, len(y_test)):
        if y_test[j] <= 0.5:
            y_test[j] = 0
        else:
            y_test[j] = 1

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

    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, predictions))

    y_pred = model.predict(X_test)

    from sklearn.metrics import accuracy_score
    print("Accuracy with RF ", accuracy_score(y_test, y_pred, normalize=True))

    RF_score = accuracy_score(y_test, y_pred, normalize=True)

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

    print("[INFO] evaluating...")
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, target_names=target_names))

    print(confusion_matrix(y_test, predictions))

    y_pred = model.predict(X_test)

    print("Accuracy with DT ", accuracy_score(y_test, y_pred, normalize=True))

    DT_score = accuracy_score(y_test, y_pred, normalize=True)

    row_contents = [str(i), str(DT_score), str(RF_score)]
    append_list_as_row(csvFileName, row_contents)


# Classification Peripheral
csvFileName = "../results/SoleModalities_GroupBased_PeripheralClassification.csv"

if not os.path.exists(csvFileName):
    with open(csvFileName, 'w', newline='') as f:
        header = ['Iteration', 'DT', 'RF']
        filewriter = csv.DictWriter(f, fieldnames=header)
        filewriter.writeheader()

list_of_lists = [first_part, second_part, third_part, fourth_part, fifth_part]
for i in range(len(list_of_lists)):
    train_lists = list_of_lists[0:i] + list_of_lists[i+1:]
    test_list = list_of_lists[i]

    import itertools
    train = list(itertools.chain(*train_lists))

    train_data = pd.concat(train, axis=0, ignore_index=True)
    train_data = train_data.fillna(0)

    X_train = train_data.iloc[:, -5:-1]
    y_train = train_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    test_data = pd.concat(test_list, axis=0, ignore_index=True)
    test_data = test_data.fillna(0)

    X_test = test_data.iloc[:, -5:-1]
    y_test = test_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_test = scaler_X.fit_transform(X_test)
    y_test = scaler_y.fit_transform(y_test)

    for j in range(0, len(y_train)):
        if y_train[j] <= 0.5:
            y_train[j] = 0
        else:
            y_train[j] = 1

    for j in range(0, len(y_test)):
        if y_test[j] <= 0.5:
            y_test[j] = 0
        else:
            y_test[j] = 1

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

    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, predictions))

    y_pred = model.predict(X_test)

    from sklearn.metrics import accuracy_score
    print("Accuracy with RF ", accuracy_score(y_test, y_pred, normalize=True))

    RF_score = accuracy_score(y_test, y_pred, normalize=True)

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

    print("[INFO] evaluating...")
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, target_names=target_names))

    print(confusion_matrix(y_test, predictions))

    y_pred = model.predict(X_test)

    print("Accuracy with DT ", accuracy_score(y_test, y_pred, normalize=True))

    DT_score = accuracy_score(y_test, y_pred, normalize=True)

    row_contents = [str(i), str(DT_score), str(RF_score)]
    append_list_as_row(csvFileName, row_contents)


# Regression Peripheral
csvFileName = "../results/SoleModalities_GroupBased_PeripheralRegression.csv"

if not os.path.exists(csvFileName):
    with open(csvFileName, 'w', newline='') as f:
        header = ['Iteration', 'LinearRegression', 'RandomForestRegressor']
        filewriter = csv.DictWriter(f, fieldnames=header)
        filewriter.writeheader()

list_of_lists = [first_part, second_part, third_part, fourth_part, fifth_part]
for i in range(len(list_of_lists)):
    train_lists = list_of_lists[0:i] + list_of_lists[i+1:]
    test_list = list_of_lists[i]

    import itertools
    train = list(itertools.chain(*train_lists))

    train_data = pd.concat(train, axis=0, ignore_index=True)
    train_data = train_data.fillna(0)

    X_train = train_data.iloc[:, -5:-1]
    y_train = train_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    test_data = pd.concat(test_list, axis=0, ignore_index=True)
    test_data = test_data.fillna(0)

    X_test = test_data.iloc[:, -5:-1]
    y_test = test_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_test = scaler_X.fit_transform(X_test)
    y_test = scaler_y.fit_transform(y_test)

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", type=str, default="aligned",
                    help="path to directory containing the 'emotions' dataset")
    ap.add_argument("-m", "--model", type=str, default="linear_regression",
                    help="type of python machine learning model to use")
    args = vars(ap.parse_args())

    models = {
        "linear_regression": LinearRegression(),
        "random_forest_regressor": RandomForestRegressor(n_estimators=120)
    }

    print("[INFO] using '{}' model".format(args["model"]))
    model = models[args["model"]]
    model.fit(X_train, y_train)

    print("[INFO] evaluating...")
    y_pred = model.predict(X_test)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error: {:.4f}".format(mse))
    print("Mean Absolute Error: {:.4f}".format(mae))
    print("R-squared: {:.4f}".format(r2))

    row_contents = [str(i), str(mae), str(mse)]
    append_list_as_row(csvFileName, row_contents)

