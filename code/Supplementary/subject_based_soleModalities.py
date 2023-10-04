import pandas as pd
import glob
import os
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from numpy import sqrt
from sklearn.metrics import confusion_matrix
from csv import writer

def nrmse(rmse, y_test): 
    nrmse = rmse / (max(y_test) - min(y_test))
    return nrmse[0]

# Append data to CSV
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

# Define paths and filenames
csvFileName = "../results/SoleModalities_SubjectBased_EEGRegression.csv"
data_path = '' #use your path

# Check if CSV file exists and create it if not
if not os.path.exists(csvFileName):
    with open(csvFileName, 'w', newline='') as f:
        header = ['Iteration', 'LR_RMSE', 'LR_NRMSE', 'DT_RMSE', 'DT_NRMSE'] 
        filewriter = writer(f, fieldnames=header)
        filewriter.writeheader()

print("Reading data...")

# Load data from multiple CSV files
all_files = glob.glob(os.path.join(data_path, "*.csv"))
li = []

for filename in all_files:
    df = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
    li.append(df)

dataset = pd.concat(li, axis=0, ignore_index=True)
dataset = dataset.fillna(0)

print("Evaluating...")

# Extract features and labels
X = dataset.iloc[:, dataset.columns != 'LABEL_SR_Arousal']
y = dataset.iloc[:, dataset.columns == 'LABEL_SR_Arousal'].values

print("X shape:", X.shape)
print("y shape:", y.shape)

# Shuffle participants list
participants_list = list(range(1, 154))
random.shuffle(participants_list)

# Regression EEG
for normloop in range(0, 153):
    participant = participants_list[normloop]
    flag = False
    fileReader = []
    increament = 0
    
    for filename in all_files:
        partNo = str(participant).zfill(3)
        if partNo in filename:
            reader = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
            lines = len(reader)
            increament += lines
            fileReader.append(reader)
            flag = True
    
    if flag:
        print("Evaluating for participant " + str(normloop) + "...")
        full_list = fileReader
        test_reader = full_list
        full_list = []
        test_data = pd.concat(test_reader, axis=0, ignore_index=True)
        test_data = test_data.dropna()
        test_X = test_data.iloc[:, :256]
        test_X = np.array(test_X)
        test_y = test_data.iloc[:, test_data.columns == 'LABEL_SR_Arousal'].values
        scaler_X2 = StandardScaler()
        scaler_y2 = StandardScaler()
        test_X = scaler_X2.fit_transform(test_X)
        test_y = scaler_y2.fit_transform(test_y)
        
        LR_list = []
        DT_list = []
        NLR_list = []
        NDT_list = []
        
        kfold = KFold(3)
        
        for train_index, test_index in kfold.split(test_X, test_y):
            X_train, X_test = test_X[train_index], test_X[test_index]
            y_train, y_test = test_y[train_index], test_y[test_index]
            
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
                
            DT_list.append(DT_rmse)
            LR_list.append(Linear_rmse)
            NDT_list.append(NDT_rmse)
            NLR_list.append(NLinear_rmse)
        
        row_contents = [str(normloop), str(np.mean(LR_list)), str(np.mean(NLR_list)), str(np.mean(DT_list)), str(np.mean(NDT_list))]
        append_list_as_row(csvFileName, row_contents)
    else:
        print("Participant does not exist")

# Regression Peripheral
csvFileName = "../results/SoleModalities_SubjectBased_PeripheralRegression.csv"

if not os.path.exists(csvFileName):
    with open(csvFileName, 'w', newline='') as f:
        header = ['Iteration', 'LR_RMSE', 'LR_NRMSE', 'DT_RMSE', 'DT_NRMSE'] 
        filewriter = writer(f, fieldnames=header)
        filewriter.writeheader()

participants_list = list(range(1, 154))
random.shuffle(participants_list)

for normloop in range(0, 153):
    participant = participants_list[normloop]
    flag = False
    fileReader = []
    increament = 0
    
    for filename in all_files:
        partNo = str(participant).zfill(3)
        if partNo in filename:
            reader = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
            lines = len(reader)
            increament += lines
            fileReader.append(reader)
            flag = True
    
    if flag:
        print("Evaluating for participant " + str(normloop) + "...")
        full_list = fileReader
        test_reader = full_list
        full_list = []
        test_data = pd.concat(test_reader, axis=0, ignore_index=True)
        test_data = test_data.dropna()
        test_X = test_data.iloc[:, -5:-1]
        test_X = np.array(test_X)
        test_y = test_data.iloc[:, test_data.columns == 'LABEL_SR_Arousal'].values
        scaler_X2 = StandardScaler()
        scaler_y2 = StandardScaler()
        test_X = scaler_X2.fit_transform(test_X)
        test_y = scaler_y2.fit_transform(test_y)
        
        LR_list = []
        DT_list = []
        NLR_list = []
        NDT_list = []
        
        kfold = KFold(3)
        
        for train_index, test_index in kfold.split(test_X, test_y):
            X_train, X_test = test_X[train_index], test_X[test_index]
            y_train, y_test = test_y[train_index], test_y[test_index]
            
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
                
            DT_list.append(DT_rmse)
            LR_list.append(Linear_rmse)
            NDT_list.append(NDT_rmse)
            NLR_list.append(NLinear_rmse)
        
        row_contents = [str(normloop), str(np.mean(LR_list)), str(np.mean(NLR_list)), str(np.mean(DT_list)), str(np.mean(NDT_list))]
        append_list_as_row(csvFileName, row_contents)
    else:
        print("Participant does not exist")

# Classification EEG
csvFileName = "../results/SoleModalities_SubjectBased_EEGClassification.csv"

if not os.path.exists(csvFileName):
    with open(csvFileName, 'w', newline='') as f:
        header = ['Iteration', 'DT', 'RF'] 
        filewriter = writer(f, fieldnames=header)
        filewriter.writeheader()

participants_list = list(range(1, 154))
random.shuffle(participants_list)

for normloop in range(0, 153):
    participant = participants_list[normloop]
    flag = False
    fileReader = []
    increament = 0
    
    for filename in all_files:
        partNo = str(participant).zfill(3)
        if partNo in filename:
            reader = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
            lines = len(reader)
            increament += lines
            fileReader.append(reader)
            flag = True
    
    if flag:
        print("Evaluating for participant " + str(normloop) + "...")
        full_list = fileReader
        test_reader = full_list
        full_list = []
        test_data = pd.concat(test_reader, axis=0, ignore_index=True)
        test_data = test_data.dropna()
        test_X = test_data.iloc[:, :256]
        test_X = np.array(test_X)
        test_y = test_data.iloc[:, test_data.columns == 'LABEL_SR_Arousal'].values
        scaler_X2 = StandardScaler()
        scaler_y2 = StandardScaler()
        test_X = scaler_X2.fit_transform(test_X)
        test_y = scaler_y2.fit_transform(test_y)
        
        RF_list = []
        DT_list = []
        
        kfold = StratifiedKFold(3)
        
        for train_index, test_index in kfold.split(test_X, test_y):
            X_train, X_test = test_X[train_index], test_X[test_index]
            y_train, y_test = test_y[train_index], test_y[test_index]
            
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
            
            model = models["random_forest"]
            model.fit(X_train, y_train)
            
            target_names = ['0=low', '1=high']
            print("[INFO] evaluating with RF...")
            predictions = model.predict(X_test)
            print(classification_report(y_test, predictions, target_names=target_names))
            print(confusion_matrix(y_test, predictions))
            
            y_pred = model.predict(X_test)
            
            print("Accuracy with RF: ", accuracy_score(y_test, y_pred, normalize=True))
            
            RF_list.append(accuracy_score(y_test, y_pred, normalize=True))
            
            model = models["decision_tree"]
            model.fit(X_train, y_train)
            
            print("[INFO] evaluating with DT...")
            predictions = model.predict(X_test)
            print(classification_report(y_test, predictions, target_names=target_names))
            print(confusion_matrix(y_test, predictions))
            
            y_pred = model.predict(X_test)
            
            print("Accuracy with DT: ", accuracy_score(y_test, y_pred, normalize=True))
            
            DT_list.append(accuracy_score(y_test, y_pred, normalize=True))
                
        DT_avg = np.mean(DT_list)
        RF_avg = np.mean(RF_list)
        
        row_contents = [str(normloop), str(DT_avg), str(RF_avg)]
        append_list_as_row(csvFileName, row_contents)
    else:
        print("Participant does not exist")

# Classification Peripheral
csvFileName = "../results/SoleModalities_SubjectBased_PeripheralClassification.csv"

if not os.path.exists(csvFileName):
    with open(csvFileName, 'w', newline='') as f:
        header = ['Iteration', 'DT', 'RF'] 
        filewriter = writer(f, fieldnames=header)
        filewriter.writeheader()

participants_list = list(range(1, 154))
random.shuffle(participants_list)

for normloop in range(0, 153):
    participant = participants_list[normloop]
    flag = False
    fileReader = []
    increament = 0
    
    for filename in all_files:
        partNo = str(participant).zfill(3)
        if partNo in filename:
            reader = pd.read_csv(filename, encoding='ISO-8859-1', header=0)
            lines = len(reader)
            increament += lines
            fileReader.append(reader)
            flag = True
    
    if flag:
        print("Evaluating for participant " + str(normloop) + "...")
        full_list = fileReader
        test_reader = full_list
        full_list = []
        test_data = pd.concat(test_reader, axis=0, ignore_index=True)
        test_data = test_data.dropna()
        test_X = test_data.iloc[:, -5:-1]
        test_X = np.array(test_X)
        test_y = test_data.iloc[:, test_data.columns == 'LABEL_SR_Arousal']
        
        scaler_X2 = StandardScaler()
        scaler_y2 = StandardScaler()
        test_X = scaler_X2.fit_transform(test_X)
        test_y = scaler_y2.fit_transform(test_y)
        
        RF_list = []
        DT_list = []
        
        kfold = StratifiedKFold(3)
        
        for train_index, test_index in kfold.split(test_X, test_y):
            X_train, X_test = test_X[train_index], test_X[test_index]
            y_train, y_test = test_y[train_index], test_y[test_index]
            
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
            
            model = models["random_forest"]
            model.fit(X_train, y_train)
            
            target_names = ['0=low', '1=high']
            print("[INFO] evaluating with RF...")
            predictions = model.predict(X_test)
            print(classification_report(y_test, predictions, target_names=target_names))
            print(confusion_matrix(y_test, predictions))
            
            y_pred = model.predict(X_test)
            
            print("Accuracy with RF: ", accuracy_score(y_test, y_pred, normalize=True))
            
            RF_list.append(accuracy_score(y_test, y_pred, normalize=True))
            
            model = models["decision_tree"]
            model.fit(X_train, y_train)
            
            print("[INFO] evaluating with DT...")
            predictions = model.predict(X_test)
            print(classification_report(y_test, predictions, target_names=target_names))
            print(confusion_matrix(y_test, predictions))
            
            y_pred = model.predict(X_test)
            
            print("Accuracy with DT: ", accuracy_score(y_test, y_pred, normalize=True))
            
            DT_list.append(accuracy_score(y_test, y_pred, normalize=True))
                
        DT_avg = np.mean(DT_list)
        RF_avg = np.mean(RF_list)
        
        row_contents = [str(normloop), str(DT_avg), str(RF_avg)]
        append_list_as_row(csvFileName, row_contents)
    else:
        print("Participant does not exist")
