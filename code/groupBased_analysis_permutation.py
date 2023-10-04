# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:08:12 2022
@author: harisushehu
"""

import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(sess)

def nrmse(rmse, y_test):
    nrmse = rmse / (max(y_test) - min(y_test))
    return nrmse[0]

print("Reading data...")

#replace with your path
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
#replace with your path
path = ''
all_files = sorted(glob.glob(path + "/*.csv"))

full_list = []

import random

participants_list = []

for k in range(1, 154):
    participants_list.append(k)

random.shuffle(participants_list)

DT_NRMSE_all = []
LR_NRMSE_all = []
Random_NRMSE_all = []

# Start for 1000 iterations
for z in range(0, 1000):

    print("Iteration", z)

    Random_list = []
    DT_list = []
    LR_list = []

    list_of_lists = [participants_list[0:29], participants_list[29:58], participants_list[58:87], participants_list[87:116], participants_list[116:]]
    for i in range(len(list_of_lists)):

        train_list = list_of_lists[0:i] + list_of_lists[i+1:]
        test_list = list_of_lists[i]

        train_data = pd.concat([li[k - 1] for k in train_list], axis=0, ignore_index=True)
        train_data = train_data.fillna(0)
        X_train = train_data.iloc[:, df.columns != 'LABEL_SR_Arousal']
        y_train = train_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

        # Scale train
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        y_train = scaler_y.fit_transform(y_train)

        test_data = pd.concat([li[k - 1] for k in test_list], axis=0, ignore_index=True)
        test_data = test_data.fillna(0)
        X_test = test_data.iloc[:, df.columns != 'LABEL_SR_Arousal']
        y_test = test_data.iloc[:, df.columns == 'LABEL_SR_Arousal'].values

        # Scale test
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_test = scaler_X.fit_transform(X_test)
        y_test = scaler_y.fit_transform(y_test)

        # Shuffle y_test values
        shuffled_y_test = y_test.copy()
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

    Random_NRMSE_all.append(np.mean(Random_list))

# ---------------------------------***Permutation test***---------------------------------------------->

countDT = 0
countLR = 0

for k in range(0, len(Random_NRMSE_all)):
    if DT_NRMSE_all > Random_NRMSE_all[k]:
        countDT += 1

    if LR_NRMSE_all > Random_NRMSE_all[k]:
        countLR += 1

permutationDT = countDT / len(Random_NRMSE_all)
permutationLR = countLR / len(Random_NRMSE_all)

alpha = 0.05

if permutationDT > alpha:
    print("DT results are not significant. p =", permutationDT)
else:
    print("DT results are significant. p =", permutationDT)

if permutationLR > alpha:
    print("LR results are not significant. p =", permutationLR)
else:
    print("LR results are significant. p =", permutationLR)
