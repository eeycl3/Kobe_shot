import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split

trainingSet = pd.read_csv('trainingSet.csv', sep=',')
y = trainingSet["shot_made_flag"]
dropColumn = ['game_date_DT', 'secondsFromPeriodStart', 'shot_made_flag', 'shot_id']
trainingSet.drop(dropColumn, axis=1, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(data_norm, y, test_size=0.33, random_state=42)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)

# RBF SVM
max_score, max_c, max_gamma = 0, 0, 0
c = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
gamma = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
for i in c:
    print("Hello from loop 1")
    for j in gamma:
        print("Hello from loop 2")
        clf_svm = SVC(C = i, gamma=j)
        scores = cross_val_score(clf_svm, X_train_norm, y_train, cv=5)
        score = scores.mean()
        if score > max_score:
            max_score = score
            max_c = i
            max_gamma = j
print("max_c: ", max_c, " max_gamma: ", max_gamma)

