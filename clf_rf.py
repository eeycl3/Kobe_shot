import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

X_train = pd.read_csv('X_train_rfe_40.csv', sep=',')
y = pd.read_csv('y_train.csv', sep = ',')
y_train = y["shot_made_flag"]

max_score = 0
min_score_std = 999999999999999999999999
param = 30
matrix_mean_score = [[0 for i in range(param)] for i in range(param)]
matrix_score_std = [[0 for i in range(param)] for i in range(param)]
for i in range(1, param):
    for j in range(1, param):
        clf_mlp = RandomForestClassifier(n_estimators= i , max_depth=j , random_state=0)
        clf_mlp.fit(X_train, y_train)
        scores = cross_val_score(clf_mlp, X_train, y_train)
        matrix_mean_score[i][j] = scores.mean()
        matrix_score_std[i][j] = scores.std()
        if scores.mean() > max_score:
            max_score = scores.mean()
            min_score_std = scores.std()
            max_i = i
            max_j = j
            best_clf_mlp = clf_mlp
print("n_estimators: ",max_i, "max_depth: ",max_j,"score: ", max_score, "std: ", min_score_std)
print(matrix_mean_score)
