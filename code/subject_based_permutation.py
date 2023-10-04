# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:25:22 2022
@author: harisushehu
"""

import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
import os
from tensorflow.python.framework import ops
ops.reset_default_graph() 
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True   
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

tf.compat.v1.keras.backend.set_session(sess)

def nrmse(rmse, y_test): 
    nrmse = (rmse) / ((max(y_test) - min(y_test)))
    return nrmse[0]

print("Reading data...")

path = '/vol/grid-solar/sgeusers/harisushehu/EMAP/Features_TAC'  # Use your path
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
path = '/vol/grid-solar/sgeusers/harisushehu/EMAP/Features_TAC'  # Use your path
all_files = sorted(glob.glob(path + "/*.csv"))

full_list = []

DT_NRMSE_all = []
LR_NRMSE_all = []
Random_NRMSE_all = []

for z in range(0, 1000):  # 1000
    print("Iteration", z)

    NDT_nrmse = []
    NLinear_nrmse = []
    NRandom_nrmse = []

    for normloop in range(0, 154):  # 154
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

            full_list.extend(fileReader)
            test_reader = full_list
            full_list = []

            test_data = pd.concat(test_reader, axis=0, ignore_index=True)

            # Drop NA
            test_data = test_data.dropna()

            import numpy as np
            test_X = test_data.iloc[:, test_data.columns != 'LABEL_SR_Arousal']
            test_X = np.array(test_X)
            test_y = test_data.iloc[:, test_data.columns == 'LABEL_SR_Arousal'].values

            # Scale test
            scaler_X2 = StandardScaler()
            scaler_y2 = StandardScaler()

            # Normalize test X and y
            test_X = scaler_X2.fit_transform(test_X)
            test_y = scaler_y2.fit_transform(test_y)

            from sklearn.tree import DecisionTreeRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error
            from sklearn.model_selection import KFold
            from numpy import sqrt
            import random

            LR_list = []
            DT_list = []
            Random_list = []
            NRandom_nrmse = []

            # Prepare cross-validation
            kfold = KFold(3)
            # Enumerate splits
            for train_index, test_index in kfold.split(test_X, test_y):
                X_train, X_test = test_X[train_index], test_X[test_index]
                y_train, y_test = test_y[train_index], test_y[test_index]

                shuffled_y_test = y_test.copy()

                # Random label
                random.shuffle(shuffled_y_test)
                random.shuffle(shuffled_y_test)
                random.shuffle(shuffled_y_test)

                Random_mse = mean_squared_error(y_test, shuffled_y_test)
                Random_rmse = sqrt(Random_mse)

                if (max(y_test) - min(y_test)) == 0:
                    NRandom = Random_rmse / 1
                else:
                    NRandom = nrmse(Random_rmse, y_test)

                Random_list.append(NRandom)

            NRandom_nrmse.append(np.mean(Random_list))

    Random_NRMSE_all.append(NRandom_nrmse[0])

# ---------------------------------***Permutation test***---------------------------------------------->

countDT = 0
countLR = 0

DT_NRMSE_all = 0.3999
LR_NRMSE_all = 26.2334

for k in range(0, len(Random_NRMSE_all)):

    if DT_NRMSE_all > float(Random_NRMSE_all[k]):
        countDT += 1

    if LR_NRMSE_all > float(Random_NRMSE_all[k]):
        countLR += 1

permutationDT = countDT / len(Random_NRMSE_all)
permutationLR = countLR / len(Random_NRMSE_all)

alpha = 0.05

if permutationDT > alpha:
    print("DT results are not significant. p =", permutationDT)
else:
    print("DT results are significant p =", permutationDT)

if permutationLR > alpha:
    print("LR results are not significant. p =", permutationLR)
else:
    print("LR results are significant p =", permutationLR)
