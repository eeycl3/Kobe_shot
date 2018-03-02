import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
trainingSet = pd.read_csv("trainingSet.csv")
y = trainingSet["shot_made_flag"]

# drop game_date_DT, secondsFromPeriodStart, secondsFromGameStart for avoiding overfit
dropColumn = ['shot_made_flag','game_date_DT','secondsFromPeriodStart', 'secondsFromGameStart','shot_id']
for col in trainingSet:
    if 'ShotType_' in col:
        dropColumn.append(col)
trainingSet.drop(columns=dropColumn, axis=1, inplace=True)
clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators= 20, max_features="auto", max_depth=10)
clf.fit(trainingSet, y)
predictSet = pd.read_csv("predictSet.csv")
yy = predictSet["shot_made_flag"]
id = predictSet["shot_id"]
predictSet.drop(dropColumn, axis=1, inplace=True)
Rf_result = clf.predict_proba(predictSet)[:,1]
Rf_scores = cross_val_score(clf, trainingSet, y)
print("Accuracy: %0.2f (+/- %0.2f)" % (Rf_scores.mean(), Rf_scores.std() * 2))

file1 = open("RF_result.csv",'w',newline='')    # goal
writer1 = csv.writer(file1)
writer1.writerow(["shot_id", "shot_made_flag"])
for index in range(len(Rf_result)):
    writer1.writerow([id[index], Rf_result[index]])

from sklearn.ensemble import AdaBoostClassifier
clf_Adaboost = AdaBoostClassifier(clf)
clf_Adaboost.fit(trainingSet, y)
ada_result = clf_Adaboost.predict_proba(predictSet)[:,1]
ada_scores = cross_val_score(clf_Adaboost, trainingSet, y)
print("Accuracy: %0.2f (+/- %0.2f)" % (ada_scores.mean(), ada_scores.std() * 2))

file2 = open("ADA_result.csv",'w',newline='')    # goal
writer2 = csv.writer(file2)
writer2.writerow(["shot_id", "shot_made_flag"])
for index in range(len(Rf_result)):
    writer2.writerow([id[index], ada_result[index]])

