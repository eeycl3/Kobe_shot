import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import RFE


trainingSet = pd.read_csv("trainingSet.csv")
y = trainingSet["shot_made_flag"]
dropColumn = ['shot_made_flag','game_date_DT','secondsFromPeriodStart','shot_id']
trainingSet.drop(columns=dropColumn, axis=1, inplace=True)


estimator = LogisticRegression()
selector = RFE(estimator, 30, step = 1)
selector = selector.fit(trainingSet, y)
X_rfe = selector.transform(trainingSet)

clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators= 20, max_features="auto", max_depth=10)
clf.fit(X_rfe, y)
kf = KFold(n_splits = 8, shuffle = True)
Rf_scores = cross_val_score(clf, X_rfe, y, cv = kf)
print("Accuracy: %0.3f (+/- %0.3f)" % (Rf_scores.mean(), Rf_scores.std() * 2))


predictSet = pd.read_csv("predictSet.csv")
yy = predictSet["shot_made_flag"]
id = predictSet["shot_id"]
predictSet.drop(dropColumn, axis=1, inplace=True)
X_pred_rfe = selector.transform(predictSet)
RFE_Rf_result = clf.predict_proba(X_pred_rfe)[:,1]


id = pd.DataFrame(id)
RFE_Rf_result = pd.DataFrame(RFE_Rf_result)
RFE_Rf_submission = pd.concat([id, RFE_Rf_result], axis = 1)
RFE_Rf_submission.to_csv('RFE_Rf_submission.csv', header=["shot_id", "shot_made_flag"], index=None)