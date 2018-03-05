import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import RFE


trainingSet = pd.read_csv("trainingSet.csv")
y = trainingSet["shot_made_flag"]
dropColumn = ['shot_made_flag','game_date_DT','secondsFromPeriodStart','shot_id']
trainingSet.drop(columns=dropColumn, axis=1, inplace=True)


estimator = LogisticRegression()
selector = RFE(estimator, 20, step = 1)
selector = selector.fit(trainingSet, y)
X_Rfe = selector.transform(trainingSet)

clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators= 20, max_features="auto", max_depth=10)
clf.fit(X_Rfe, y)
kf = KFold(n_splits = 8, shuffle = True)
Rfe_Rf_scores = cross_val_score(clf, X_Rfe, y, cv = kf)
print("Accuracy: %0.3f (+/- %0.3f)" % (Rfe_Rf_scores.mean(), Rfe_Rf_scores.std() * 2))


predictSet = pd.read_csv("predictSet.csv")
#yy = predictSet["shot_made_flag"]
id = predictSet["shot_id"]
predictSet.drop(dropColumn, axis=1, inplace=True)
X_pred_Rfe = selector.transform(predictSet)
Rfe_Rf_result = clf.predict_proba(X_pred_Rfe)[:,1]


id = pd.DataFrame(id)
Rfe_Rf_result = pd.DataFrame(Rfe_Rf_result)
Rfe_Rf_submission = pd.concat([id, Rfe_Rf_result], axis = 1)
Rfe_Rf_submission.to_csv('Rfe_Rf_submission.csv', header=["shot_id", "shot_made_flag"], index=None)