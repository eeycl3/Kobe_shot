import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
trainingSet = pd.read_csv("trainingSet.csv")
y = trainingSet["shot_made_flag"]

# drop columns 
dropColumn = ['shot_made_flag','game_date_DT','secondsFromPeriodStart','shot_id']
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
kf = KFold(n_splits = 8, shuffle = True)
Rf_scores = cross_val_score(clf, trainingSet, y, cv = kf)
print("Accuracy: %0.3f (+/- %0.3f)" % (Rf_scores.mean(), Rf_scores.std() * 2))

id = pd.DataFrame(id)
Rf_result = pd.DataFrame(Rf_result)
Rf_submission = pd.concat([id, Rf_result], axis = 1)
Rf_submission.to_csv('Rf_submission.csv', header=["shot_id", "shot_made_flag"], index=None)

from sklearn.ensemble import AdaBoostClassifier
clf_Adaboost = AdaBoostClassifier(clf)
clf_Adaboost.fit(trainingSet, y)
ada_result = clf_Adaboost.predict_proba(predictSet)[:,1]
ada_scores = cross_val_score(clf_Adaboost, trainingSet, y)
print("Accuracy: %0.2f (+/- %0.2f)" % (ada_scores.mean(), ada_scores.std() * 2))

id = pd.DataFrame(id)
ada_result = pd.DataFrame(ada_result)
Rf_submission = pd.concat([id, ada_result], axis = 1)
Rf_submission.to_csv('ada_submission.csv', header=["shot_id", "shot_made_flag"], index=None)
