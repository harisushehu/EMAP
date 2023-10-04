# Accessing the EMAP Dataset

Researchers who wish to use the Emotional Arousal Pattern (EMAP) dataset can request access to the dataset from [here](https://www.wgtn.ac.nz/psyc/research/emap-open-database).

# EMAP Dataset Paper Code Repository

Welcome to the EMAP (Emotional Arousal Pattern) dataset paper code repository. This repository contains code to support the research presented in our paper titled:

**Eisenbarth, H., Oxner, M., Shehu, H. A., Gastrell, T., Walsh, A., Browne, W. N., & Xue, B. (2023).**  
**Emotional Arousal Pattern (EMAP): A New Database for Modeling Momentary Subjective and Psychophysiological Responses to Affective Stimuli.**  
*Psychophysiology, e14446.*

## MATLAB Code

### 'MATLAB Scripts - EMAP Raw Dataset'
- Extracts the raw datasets from the EEGLab set.

### 'MATLAB Scripts - EMAP Clean Dataset from Raw Dataset'
- Extracts the clean dataset in CSV format from the raw data.

## Python Code

This repository includes various Python scripts to perform different analyses related to the EMAP dataset. Below are descriptions of each code file:

1. **feature_extraction.py**  
   - Extract EEG features (Alpha, Beta, Gamma, and Theta channels) from pre-processed EEG files.

2. **feature_correlation.py**  
   - Calculate the correlation between arousal ratings, EEG data, and peripheral measures (e.g., heart rate, skin conductance, blood volume, respiration).

3. **feature_selection_LR_shift0.py** and **feature_selection_DT_shift0.py**  
   - Determine the minimum number of features contributing to predicting arousal ratings using Linear Regression and Decision Tree, respectively, with no temporal shift.

4. **feature_selection_LR_shift5.py** and **feature_selection_DT_shift5.py**  
   - Obtain the minimum number of features contributing to predicting arousal ratings using Linear Regression and Decision Tree, respectively, with a temporal shift of 2500 ms (you can adjust the shift in the code).

5. **feature_selection_DT_classification.py** and **feature_selection_RF_classification.py**  
   - Identify the minimum number of features essential for categorizing arousal ratings using Decision Tree and Random Forests, respectively.

6. **group_based_categorization.py**, **group_based_categorization_mostArousedSegment.py**, **group_based_categorization_overallRatings.py**, and **group_based_categorization_random.py**  
   - Perform group-based categorization of the originally extracted data, most emotionally aroused data segments, overall ratings, and random labels.

7. **group_based_regression_mostArousedSegment.py**, **group_based_regression_overall_rating.py**, and **groupBased_analysis_permutation.py**  
   - Conduct group-based regression using the most emotionally aroused data segments, overall ratings, and perform group-based permutation tests.

8. **subject_based_regression.py**, **subject_based_regression_mostArousedSegment.py**, **subject_based_regression_overallRatings.py**, **subject_based_permutation.py**  
   - Perform subject-based regression using the originally extracted data, the most emotionally aroused data segments, overall ratings, and group-based permutation tests.

9. **stats.py**  
   - Perform statistical analyses.

10. **evoked_topomaps.py**  
    - Generate topographical plots.

## Supplementary Codes

Additionally, you will find codes for:

- Predicting and categorizing arousal ratings (both within and across participants).
- Obtaining graphs to visualize the differences in predicting moment-by-moment and overall arousal ratings. These can be found in the "Supplementary" folder.

Feel free to explore and use these scripts to replicate or extend our research. If you have any questions or need assistance, please don't hesitate to contact us.

Thank you for your interest in our work!

**The EMAP Research Team**
