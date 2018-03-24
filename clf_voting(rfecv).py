
# coding: utf-8

# In[39]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


# In[40]:


trainingSet = pd.read_csv("trainingSet.csv")
y = trainingSet["shot_made_flag"]
dropColumn = ['shot_made_flag','game_date_DT','secondsFromPeriodStart','shot_id']
trainingSet.drop(columns=dropColumn, axis=1, inplace=True)

scaler = StandardScaler()
trainingSet = scaler.fit_transform(trainingSet)


# In[43]:


estimator = RandomForestClassifier()
selector = RFECV(estimator, 40)
selector = selector.fit(trainingSet, y)
X_Rfecv = selector.transform(trainingSet)


# In[45]:


clf_Rf = RandomForestClassifier(random_state=0, n_estimators= 32, max_depth=10)
clf_Rf.fit(X_Rfecv, y)
kf = KFold(n_splits = 3, shuffle = True)
Rfecv_Rf_scores = cross_val_score(clf_Rf, X_Rfecv, y, cv = kf)
print("Accuracy: %0.3f (+/- %0.3f)" % (Rfe_Rf_scores.mean(), Rfe_Rf_scores.std() * 2))


# In[46]:


svm = SVC(C=2, gamma=0.02)
clf_svm = CalibratedClassifierCV(svm)
clf_svm.fit(X_Rfecv, y)
kf = KFold(n_splits = 3, shuffle = True)
Rfecv_svm_scores = cross_val_score(clf_svm, X_Rfecv, y, cv = kf)
print("Accuracy: %0.3f (+/- %0.3f)" % (Rfe_svm_scores.mean(), Rfe_svm_scores.std() * 2))


# In[49]:


clf_GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth = 1, random_state=0)
clf_GBC.fit(X_Rfecv, y)
kf = KFold(n_splits = 3, shuffle = True)
Rfecv_GBC_scores = cross_val_score(clf_GBC, X_Rfecv, y, cv = kf)
print("Accuracy: %0.3f (+/- %0.3f)" % (Rfe_GBC_scores.mean(), Rfe_GBC_scores.std() * 2))


# In[50]:


# voting 1
eclf1 = VotingClassifier(estimators = [('Rf', clf_Rf), ('SVM', clf_svm), ('GBC', clf_GBC)], voting = 'soft')
eclf1.fit(X_Rfecv, y)
kf = KFold(n_splits = 3, shuffle = True)
eclf1_scores = cross_val_score(eclf1, X_Rfecv, y, cv = kf)
print("Accuracy: %0.3f (+/- %0.3f)" % (eclf1_scores.mean(), eclf1_scores.std() * 2))


# In[51]:


predictSet = pd.read_csv("predictSet.csv")
#yy = predictSet["shot_made_flag"]
id = predictSet["shot_id"]
predictSet.drop(dropColumn, axis=1, inplace=True)
predictSet = scaler.transform(predictSet)
X_pred_Rfecv = selector.transform(predictSet)
Rfecv_eclf1_result = eclf1.predict_proba(X_pred_Rfecv)[:,1]


id = pd.DataFrame(id)
Rfecv_eclf1_result = pd.DataFrame(Rfecv_eclf1_result)
Rfecv_eclf1_submission = pd.concat([id, Rfecv_eclf1_result], axis = 1)
Rfecv_eclf1_submission.to_csv('rfecv40_eclf1_submission.csv', header=["shot_id", "shot_made_flag"], index=None)


# In[ ]:


# voting 2
eclf2 = VotingClassifier(estimators = [('rf', clf_Rf), ('GBC', clf_GBC)], voting = 'soft')
eclf2.fit(X_Rfecv, y)
kf = KFold(n_splits = 3, shuffle = True)
eclf2_scores = cross_val_score(eclf2, X_Rfecv, y, cv = kf)
print("Accuracy: %0.3f (+/- %0.3f)" % (eclf2_scores.mean(), eclf2_scores.std() * 2))


# In[ ]:


predictSet = pd.read_csv("predictSet.csv")
#yy = predictSet["shot_made_flag"]
id = predictSet["shot_id"]
predictSet.drop(dropColumn, axis=1, inplace=True)
predictSet = scaler.transform(predictSet)
X_pred_Rfecv = selector.transform(predictSet)
Rfecv_eclf2_result = eclf2.predict_proba(X_pred_Rfecv)[:,1]


id = pd.DataFrame(id)
Rfecv_eclf2_result = pd.DataFrame(Rfecv_eclf2_result)
Rfecv_eclf2_submission = pd.concat([id, Rfecv_eclf2_result], axis = 1)
Rfecv_eclf2_submission.to_csv('rfecv40_eclf2_submission.csv', header=["shot_id", "shot_made_flag"], index=None)

