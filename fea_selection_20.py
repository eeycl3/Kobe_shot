import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler


#prepare dataset
trainingSet = pd.read_csv("trainingSet.csv")
y = trainingSet["shot_made_flag"]
dropColumn = ['shot_made_flag','game_date_DT','secondsFromPeriodStart','shot_id']
trainingSet.drop(columns=dropColumn, axis=1, inplace=True)

#train test split
X_train, X_test, y_train, y_test = train_test_split(trainingSet, y, test_size=0.33, random_state=42)

#z-score normalization
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)


#recursive feature selection
estimator = LogisticRegression()
selector = RFE(estimator, 20, step = 1)
selector = selector.fit(X_train_norm, y_train)
X_train_rfe = selector.transform(X_train_norm)
X_test_rfe = selector.transform(X_test_norm)

X_train_rfe = pd.DataFrame(X_train_rfe)
X_test_rfe = pd.DataFrame(X_test_rfe)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)


#get the selected features
tf_list = selector.support_.tolist()
original_header = list(trainingSet)
new_header = list()
for i in range(len(original_header)):
    if tf_list[i]:
        new_header.append(original_header[i])

#rename header of dataset
X_train_rfe.columns = new_header
X_test_rfe.columns = new_header
#to csv
X_train_rfe.to_csv("X_train_rfe_20.csv", index = None)
X_test_rfe.to_csv("X_test_rfe_20.csv", index = None)
y_train.to_csv("y_train_rfe_20.csv", index=None)
y_test.to_csv("y_test_rfe_20.csv", index=None)