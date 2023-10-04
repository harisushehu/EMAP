# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:45:00 2022
@author: harisushehu
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import glob
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.python.framework import ops
ops.reset_default_graph() 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
tf.compat.v1.keras.backend.set_session(sess)

# Define model
def model(X, dropout):
    # Initialize RNN
    regressor = Sequential()
    # Add first layer
    regressor.add(LSTM(units=150, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1)))
    regressor.add(Dropout(dropout))  # dropout = 0.2
    regressor.add(Dense(units=150))
    regressor.add(Dropout(dropout))
    # Add fourth layer
    regressor.add(LSTM(units=75))
    regressor.add(Dropout(dropout))
    # Add fifth layer
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return regressor

def nrmse(rmse, y_test):
    nrmse = (rmse) / ((max(y_test) - min(y_test)))
    return nrmse[0]

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # Getting the name of the column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]  # Deleting the column from the dataset

# Save results
import csv
from csv import writer

# Append data in CSV function
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

csv_filename = "./results_GSR_SubjectBasedByParticipant_Graph.csv"

# Read in CSV file
if os.path.exists(csv_filename):
    print()
else:
    with open(csv_filename, 'w', newline='') as f:
        header = ['Fold', 'Participant', 'RMSE', 'NRMSE', 'LRRMSE', 'LRNRMSE', 'DTRMSE', 'DTNRMSE']
        filewriter = csv.DictWriter(f, fieldnames=header)
        filewriter.writeheader()

print("Reading data...")

path = '/vol/grid-solar/sgeusers/harisushehu/EMAP/Features'
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
    li.append(df)

dataset = pd.concat(li, axis=0, ignore_index=True)

# Drop NaN (missing) features
dataset = dataset.dropna()

print("Evaluating...")

X = dataset.iloc[:, df.columns != 'LABEL_SR_Arousal']
y = dataset.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

print("X is:", X.shape)
print("y is:", y.shape)

print("Reading data for shifting and normalization by participants...")
path = '/vol/grid-solar/sgeusers/harisushehu/EMAP/Features'
all_files = sorted(glob.glob(path + "/*.csv"))

psysio = []
label = []

numInit = 0
for normloop in range(1, 154):  # 154
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
        data = pd.concat(fileReader, axis=0, ignore_index=True)
        # Dropping NaN (missing) features
        data = data.dropna()
        X = data.iloc[:, data.columns != 'LABEL_SR_Arousal']
        y = data.iloc[:, data.columns == 'LABEL_SR_Arousal'].values
        # Removing highly correlated features
        # correlation(X, 0.99)
        print(X.shape)
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        keras_callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-3, mode='min'),
            ModelCheckpoint("best_arousal_model.hdf5", monitor='val_loss', save_best_only=True, mode='min',
                            save_freq='epoch')]
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # Normalize X and y
        X_train = sc_X.fit_transform(X_train)
        y_train = sc_y.fit_transform(y_train)
        sc1_X = StandardScaler()
        sc1_y = StandardScaler()
        X_test = sc1_X.fit_transform(X_test)
        y_test = sc1_y.fit_transform(y_test)
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.linear_model import LinearRegression
        LinearReg = LinearRegression().fit(X_train, y_train)
        DTReg = DecisionTreeRegressor(random_state=1).fit(X_train, y_train)
        LinearPred = LinearReg.predict(X_test)
        DTPred = DTReg.predict(X_test)
        Linear_mse = mean_squared_error(y_test, LinearPred)
        Linear_rmse = sqrt(Linear_mse)
        DT_mse = mean_squared_error(y_test, DTPred)
        DT_rmse = sqrt(DT_mse)
        print("****************************************************")
        print("Results for participant no. " + str(normloop))
        # Evaluate Performance
        print("LR Mean squared error", Linear_mse)
        print("LR Root mean squared error", Linear_rmse)
        NLinear_nrmse = nrmse(Linear_rmse, y_test)
        print("LR Normalized root mean squared error", NLinear_nrmse)
        print("****************************************************")
        print("Results for participant no. " + str(normloop))
        # Evaluate Performance
        print("DT Mean squared error", DT_mse)
        print("DT Root mean squared error", DT_rmse)
        NDT_nrmse = nrmse(DT_rmse, y_test)
        print("DT Normalized root mean squared error", NDT_nrmse)
        from matplotlib import pyplot as plt
        predictions = DTReg.predict(X_test)
        predictions = sc1_y.inverse_transform(predictions)
        train, test = data[-len(X_train):], data[0:-len(X_train)]
        valid = data[len(X_train):]
        valid['Predictions'] = predictions
        plt.clf()
        f, ax = plt.subplots(1)
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('Arousal', fontsize=10)
        plt.plot(valid[['LABEL_SR_Arousal', 'Predictions']])
        plt.legend(['Test', 'Predictions'], loc='lower right')
        ax.set_ylim(ymin=-0.5e-10, ymax=1.5)
        ax.set_xlim(xmin=0, xmax=3000)
        filename = "./Predictions/Pred_" + str(normloop)
        plt.savefig(filename)
    else:
        print("Participant does not exist")
