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


X_train, X_test, y_train, y_test = train_test_split(trainingSet, y, test_size=0.33, random_state=42)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_train_norm = scaler.fit_transform(X_train)

X_train_rfe = pd.read_csv('X_train_rfe_20.csv', sep=',')
X_test_rfe = pd.read_csv('X_test_rfe_20.csv', sep=',')


# RBF SVM
max_score, max_c, max_gamma = 0, 0, 0
c = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
sigma = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
gamma = [1/(2.0 * i * i) for i in sigma]
scores_mean = list()
for i in c:
    for j in gamma: 
        clf_svm = SVC(C = i, gamma=j)
        scores = cross_val_score(clf_svm, X_train_rfe, y_train, cv=5)
        score = scores.mean()
        scores_mean.append(score)
        if score > max_score:
            max_score = score
            max_c = i
            max_gamma = j
print("max_c: ", max_c, " max_gamma: ", max_gamma)



