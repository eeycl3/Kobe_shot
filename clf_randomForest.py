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
clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators= 15, max_features="auto", max_depth=10)
clf.fit(trainingSet, y)
predictSet = pd.read_csv("predictSet.csv")
yy = predictSet["shot_made_flag"]
id = predictSet["shot_id"]
predictSet.drop(dropColumn, axis=1, inplace=True)
result = clf.predict_proba(predictSet)[:,1]
scores = cross_val_score(clf, trainingSet, y)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

file1 = open("RF_result.csv",'w',newline='')    # goal
writer1 = csv.writer(file1)
writer1.writerow(["shot_id", "shot_made_flag"])
for index in range(len(result)):
    writer1.writerow([id[index], result[index]])

