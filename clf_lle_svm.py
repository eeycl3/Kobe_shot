import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.manifold import LocallyLinearEmbedding

X_train = pd.read_csv('X_train_rfe_20.csv', sep=',')
X_test = pd.read_csv('X_test_rfe_20.csv', sep=',')
y_train = pd.read_csv('y_train.csv', sep=',')
y_test = pd.read_csv('y_test.csv', sep=',')

lle = LocallyLinearEmbedding(n_neighbors=5, n_components=2, eigen_solver='dense')
lle_X_train = lle.fit_transform(X_train)

# RBF SVM
max_score, max_c, max_gamma = 0, 0, 0
c = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
sigma = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
gamma = [1/(2.0 * i * i) for i in sigma]
scores_mean = list()
for i in c:
    for j in gamma: 
        clf_svm = SVC(C = i, gamma=j)
        scores = cross_val_score(clf_svm, lle_X_train, y_train.values.ravel())
        score = scores.mean()
        scores_mean.append(score)
        if score > max_score:
            max_score = score
            max_c = i
            max_gamma = j
print("max_c: ", max_c, " max_gamma: ", max_gamma)