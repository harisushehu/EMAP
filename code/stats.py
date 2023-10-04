# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:26:34 2022
@author: harisushehu
"""

import pandas as pd
from numpy.random import seed
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon

# Set the random seed
seed(1)

# Load data
def load_data(filename, column_name):
    data = pd.read_csv(filename, encoding='ISO-8859-1')[column_name][0:5]
    data_list = [item for item in data]
    return data_list

DT_before = load_data("../data/results_DT_shift0.csv", 'Before_NRMSE')
DT_after = load_data("../data/results_DT_shift0.csv", 'After_NRMSE')
LR_before = load_data("../data/results_LR_shift0.csv", 'Before_NRMSE')
LR_after = load_data("../data/results_LR_shift0.csv", 'After_NRMSE')

# Wilcoxon Signed Rank Test
import numpy as np

DT_before = np.float32(DT_before)
DT_after = np.float32(DT_after)
LR_before = np.float32(LR_before)
LR_after = np.float32(LR_after)

# Compare samples DT vs LR before FS
stat, p = wilcoxon(DT_before, LR_before)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')

# Compare samples before and after FS for DT
stat, p = wilcoxon(DT_before, DT_after)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')

# Compare samples before and after FS for LR
stat, p = wilcoxon(LR_before, LR_after)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')

# ... Repeat the same structure for other comparisons
