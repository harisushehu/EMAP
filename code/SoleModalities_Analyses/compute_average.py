#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:12:33 2024

@author: harisushehu
"""

import os
import pandas as pd

# Function to calculate average and standard deviation for a given number of rows
def calculate_stats(df, num_rows):
    df_no_iteration = df.drop(columns=['Iteration'])
    avg = df_no_iteration.head(num_rows).mean()
    std_dev = df_no_iteration.head(num_rows).std()
    return avg, std_dev

# Function to process CSV files in a folder
def process_csv_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().find('group') != -1:
            rows_to_process = 5
        elif filename.lower().find('subject') != -1:
            rows_to_process = 145
        else:
            continue
        
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
            if set(['Iteration', 'LR_RMSE', 'LR_NRMSE', 'DT_RMSE', 'DT_NRMSE']).issubset(df.columns):
                avg, std_dev = calculate_stats(df, rows_to_process)
                print(f"File: {filename}")
                for column in avg.index:
                    print(f"{column}: Avg = {avg[column]:.2f} +- {std_dev[column]:.2f}")
                print()

# Specify the folder path containing CSV results
folder_path = '../results'

# Process CSV files in the folder
process_csv_files(folder_path)
