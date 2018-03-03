import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import numpy as np

trainingSet = pd.read_csv('trainingSet.csv', sep=',')
y = trainingSet["shot_made_flag"]
dropColumn = ['game_date_DT', 'secondsFromPeriodStart', 'shot_made_flag', 'shot_id']
trainingSet.drop(dropColumn, axis=1, inplace=True)


scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(trainingSet)
data_norm = scaler.transform(trainingSet)

# x_train, x_test, y_train, y_test = train_test_split(data_norm, y, test_size=0.3, random_state=42)

# svm = SVC()
# svm = SVC(kernel='linear')
svm = LinearSVC(C=0.0023)
clf = CalibratedClassifierCV(svm)
clf.fit(data_norm, y)
# svm.fit(x_train, y_train)

predictSet = pd.read_csv("predictSet.csv")
print(len(predictSet))
y_predictSet = predictSet['shot_made_flag']
p_id = predictSet['shot_id']

predictSet.drop(dropColumn, axis=1, inplace=True)

scaler_pre = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler_pre.fit(predictSet)
data_norm_pre = scaler_pre.transform(predictSet)

svm_result = clf.predict_proba(data_norm_pre)[:, 1]
svm_score = cross_val_score(clf, data_norm, y)

print("Accuracy: %0.2f (+/- %0.2f)" % (svm_score.mean(), svm_score.std() * 2))

file_svm = open("svm_result.csv", 'w', newline='')
writer_svm = csv.writer(file_svm)
writer_svm.writerow(["shot_id", "shot_made_flag"])
for row in range(len(svm_result)):
    writer_svm.writerow([p_id[row], svm_result[row]])
