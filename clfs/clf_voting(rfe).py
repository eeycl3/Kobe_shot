
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


# In[30]:


trainingSet = pd.read_csv("trainingSet.csv")
y = trainingSet["shot_made_flag"]
dropColumn = ['shot_made_flag','game_date_DT','secondsFromPeriodStart','shot_id']
trainingSet.drop(columns=dropColumn, axis=1, inplace=True)

scaler = StandardScaler()
trainingSet = scaler.fit_transform(trainingSet)


# In[31]:


estimator = RandomForestClassifier()
selector = RFE(estimator, 20, step = 1)
selector = selector.fit(trainingSet, y)
X_Rfe = selector.transform(trainingSet)


# In[32]:


clf_Rf = RandomForestClassifier(random_state=0, n_estimators= 32, max_depth=10)
clf_Rf.fit(X_Rfe, y)
kf = KFold(n_splits = 3, shuffle = True)
Rfe_Rf_scores = cross_val_score(clf_Rf, X_Rfe, y, cv = kf)
print("Accuracy: %0.3f (+/- %0.3f)" % (Rfe_Rf_scores.mean(), Rfe_Rf_scores.std() * 2))


# In[33]:


svm = SVC(C=2, gamma=0.02)
clf_svm = CalibratedClassifierCV(svm)
clf_svm.fit(X_Rfe, y)
kf = KFold(n_splits = 3, shuffle = True)
Rfe_svm_scores = cross_val_score(clf_svm, X_Rfe, y, cv = kf)
print("Accuracy: %0.3f (+/- %0.3f)" % (Rfe_svm_scores.mean(), Rfe_svm_scores.std() * 2))


# In[34]:


clf_GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth = 1, random_state=0)
clf_GBC.fit(X_Rfe, y)
kf = KFold(n_splits = 3, shuffle = True)
Rfe_GBC_scores = cross_val_score(clf_GBC, X_Rfe, y, cv = kf)
print("Accuracy: %0.3f (+/- %0.3f)" % (Rfe_GBC_scores.mean(), Rfe_GBC_scores.std() * 2))


# In[35]:


# voting 1
eclf1 = VotingClassifier(estimators = [('Rf', clf_Rf), ('SVM', clf_svm), ('GBC', clf_GBC)], voting = 'soft')
eclf1.fit(X_Rfe, y)
kf = KFold(n_splits = 3, shuffle = True)
eclf1_scores = cross_val_score(eclf1, X_Rfe, y, cv = kf)
print("Accuracy: %0.3f (+/- %0.3f)" % (eclf1_scores.mean(), eclf1_scores.std() * 2))


# In[36]:


predictSet = pd.read_csv("predictSet.csv")
#yy = predictSet["shot_made_flag"]
id = predictSet["shot_id"]
predictSet.drop(dropColumn, axis=1, inplace=True)
predictSet = scaler.transform(predictSet)
X_pred_Rfe = selector.transform(predictSet)
Rfe_eclf1_result = eclf1.predict_proba(X_pred_Rfe)[:,1]


id = pd.DataFrame(id)
Rfe_eclf1_result = pd.DataFrame(Rfe_eclf1_result)
Rfe_eclf1_submission = pd.concat([id, Rfe_eclf1_result], axis = 1)
Rfe_eclf1_submission.to_csv('rfe20_eclf1_submission.csv', header=["shot_id", "shot_made_flag"], index=None)


# In[37]:


# voting 2
eclf2 = VotingClassifier(estimators = [('SVM', clf_svm), ('GBC', clf_GBC)], voting = 'soft')
eclf2.fit(X_Rfe, y)
kf = KFold(n_splits = 3, shuffle = True)
eclf2_scores = cross_val_score(eclf2, X_Rfe, y, cv = kf)
print("Accuracy: %0.3f (+/- %0.3f)" % (eclf2_scores.mean(), eclf2_scores.std() * 2))


# In[38]:


predictSet = pd.read_csv("predictSet.csv")
#yy = predictSet["shot_made_flag"]
id = predictSet["shot_id"]
predictSet.drop(dropColumn, axis=1, inplace=True)
predictSet = scaler.transform(predictSet)
X_pred_Rfe = selector.transform(predictSet)
Rfe_eclf2_result = eclf2.predict_proba(X_pred_Rfe)[:,1]


id = pd.DataFrame(id)
Rfe_eclf2_result = pd.DataFrame(Rfe_eclf2_result)
Rfe_eclf2_submission = pd.concat([id, Rfe_eclf2_result], axis = 1)
Rfe_eclf2_submission.to_csv('rfe20_eclf2_submission.csv', header=["shot_id", "shot_made_flag"], index=None)

