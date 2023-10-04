# -*- coding: utf-8 -*-
"""
Created on Wed Aug 3 21:14:10 2022
@author: harisushehu
"""

import numpy as np
import pandas as pd
import os
from scipy import signal
from scipy.integrate import simps
from matplotlib import pyplot as plt

# Function for preprocessing and feature extraction
def preprocessing_feature_extraction(data, eeg_names, peripheral_names, label_name):
    # Define channel names
    channel_names = ["trialNumber","heartrate","GSR","IRPleth","Respir",
                     "EEG_AF3","EEG_AF4","EEG_AF7","EEG_AF8","EEG_C1","EEG_C2",
                     "EEG_C3","EEG_C4","EEG_C5","EEG_C6","EEG_CP1","EEG_CP2",
                     "EEG_CP3","EEG_CP4","EEG_CP5","EEG_CP6","EEG_CPz","EEG_Cz",
                     "EEG_F1","EEG_F2","EEG_F3","EEG_F4","EEG_F5","EEG_F6",
                     "EEG_F7","EEG_F8","EEG_FC1","EEG_FC2","EEG_FC3","EEG_FC4",
                     "EEG_FC5","EEG_FC6","EEG_Fp1","EEG_Fp2","EEG_FT10","EEG_FT7",
                     "EEG_FT8","EEG_FT9","EEG_Fz","EEG_O1","EEG_O2","EEG_Oz",
                     "EEG_P1","EEG_P2","EEG_P3","EEG_P4","EEG_P5","EEG_P6",
                     "EEG_P7","EEG_P8","EEG_PO3","EEG_PO4","EEG_PO7","EEG_PO8",
                     "EEG_POz","EEG_Pz","EEG_T7","EEG_T8","EEG_TP10","EEG_TP7",
                     "EEG_TP8","EEG_TP9","EEG_AFz", "EEG_FCz"]
    
    data_all_channels = data[channel_names]

    spectral_names = []
    spectral_names.append("trialNumber")
    
    # Create feature names based on EEG and peripheral sensors
    for name in eeg_names:
        spectral_names.append(name+"_Theta")
        spectral_names.append(name+"_Alpha")
        spectral_names.append(name+"_Beta")
        spectral_names.append(name+"_Gamma")
    for name in peripheral_names:
        spectral_names.append(name+"_mean")
        spectral_names.append(name+"_std")

    spectral_names.append(label_name)

    spectral_features = pd.DataFrame(columns=spectral_names)

    k = 0
    
    # Loop through data in chunks of 125
    for i in range(0, len(data) - len(data) % 125, 125):
        bin_data = data_all_channels[i:(i + 125)]
        bin_labels = data["contArousal"][i:(i + 125)]
        trial = data["trialNumber"][i:(i + 125)]
        row = []

        row.append(np.mean(trial))

        for j in eeg_names:
            freqs, psd = signal.welch(bin_data[j], 250, nperseg=125)
            idx_theta = np.logical_and(freqs >= 4, freqs <= 8)
            idx_alpha = np.logical_and(freqs >= 8, freqs <= 13)
            idx_beta = np.logical_and(freqs >= 13, freqs <= 30)
            idx_gamma = np.logical_and(freqs >= 30, freqs <= 60)

            freq_res = freqs[1] - freqs[0]

            row.append(simps(psd[idx_theta], dx=freq_res))
            row.append(simps(psd[idx_alpha], dx=freq_res))
            row.append(simps(psd[idx_beta], dx=freq_res))
            row.append(simps(psd[idx_gamma], dx=freq_res))

        for j in peripheral_names:
            row.append(np.mean(bin_data[j]))
            row.append(np.std(bin_data[j]))

        row.append(np.mean(bin_labels))
        spectral_features.loc[k] = row
        k += 1
    return spectral_features

# Function to get unique values from a list (integers only)
def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list and x % 1 == 0:
            unique_list.append(x)
    return unique_list

# Define paths for input and output data
data_path = "../data/Data"
output_path = "../data/FE/Features"

spectral_names = []
eeg_names = ["EEG_AF3","EEG_AF4","EEG_AF7","EEG_AF8","EEG_C1","EEG_C2","EEG_C3","EEG_C4","EEG_C5","EEG_C6","EEG_CP1","EEG_CP2","EEG_CP3","EEG_CP4","EEG_CP5","EEG_CP6","EEG_CPz","EEG_Cz","EEG_F1","EEG_F2","EEG_F3","EEG_F4","EEG_F5","EEG_F6","EEG_F7","EEG_F8","EEG_FC1","EEG_FC2","EEG_FC3","EEG_FC4","EEG_FC5","EEG_FC6","EEG_Fp1","EEG_Fp2","EEG_FT10","EEG_FT7","EEG_FT8","EEG_FT9","EEG_Fz","EEG_O1","EEG_O2","EEG_Oz","EEG_P1","EEG_P2","EEG_P3","EEG_P4","EEG_P5","EEG_P6","EEG_P7","EEG_P8","EEG_PO3","EEG_PO4","EEG_PO7","EEG_PO8","EEG_POz","EEG_Pz","EEG_T7","EEG_T8","EEG_TP10","EEG_TP7","EEG_TP8","EEG_TP9","EEG_AFz", "EEG_FCz"]
peripheral_names = ["heartrate","GSR","IRPleth","Respir"]
label_name = "LABEL_SR_Arousal"

# Create feature names based on EEG and peripheral sensors
for name in eeg_names:
    spectral_names.append(name+"_Theta")
    spectral_names.append(name+"_Alpha")
    spectral_names.append(name+"_Beta")
    spectral_names.append(name+"_Gamma")
for name in peripheral_names:
    spectral_names.append(name+"_mean")
    spectral_names.append(name+"_std")
    
spectral_names.append(label_name)

# Loop through participants
for i in range(0, 153):
    if len(str(i+1)) == 1:  
        partNo = "00" + str(i+1)
    elif len(str(i+1)) == 2:  
        partNo = "0" + str(i+1)
    else:
        partNo = str(i+1)
        
    flag = False   
    li = []
    nameList = []
    
    # Loop through files in the data path
    for filename in os.listdir(data_path):
        if(partNo in filename):
            nameList.append(filename)
            df = pd.read_csv(os.path.join(data_path, filename), index_col=None, header=0)
            li.append(df)
            flag = True
            
    if flag == True:
        frame = pd.concat(li, axis=0, ignore_index=True)
        processed_data = preprocessing_feature_extraction(frame, eeg_names, peripheral_names, label_name)
        
        for k in range(0, len(nameList)):
            data_index = unique(processed_data["trialNumber"])[k]
            dataToSave = processed_data[processed_data["trialNumber"] == data_index]
            dataToSave[spectral_names].to_csv(os.path.join(output_path,"Features_"+nameList[k]), sep = ',', index = False)
            print("Process and Exported: "+nameList[k])
