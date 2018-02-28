import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
trainingSet = pd.read_csv("trainingSet.csv")
y = trainingSet["shot_made_flag"]

# drop game_date_DT, secondsFromPeriodStart, secondsFromGameStart for avoiding overfit
dropColumn = ['shot_made_flag','game_date_DT','secondsFromPeriodStart', 'secondsFromGameStart']
for col in trainingSet:
    if 'ShotType_' in col:
        dropColumn.append(col)
trainingSet.drop(columns=dropColumn, axis=1, inplace=True)
clf = RandomForestClassifier(n_jobs=2, random_state=0, oob_score= True, n_estimators= 100, max_features="auto")
clf.fit(trainingSet, y)
scores = cross_val_score(clf, trainingSet, y)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))