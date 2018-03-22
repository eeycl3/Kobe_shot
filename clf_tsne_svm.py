import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.manifold import TSNE

X_train_20 = pd.read_csv('X_train_rfe_20.csv', sep=',')
X_test = pd.read_csv('X_test_rfe_20.csv', sep=',')
y_train = pd.read_csv('y_train.csv', sep=',')
y_test = pd.read_csv('y_test.csv', sep=',')

tsne = TSNE(n_components=2)
tsne_X_train_20 = tsne.fit_transform(X_train_20)

# RBF SVM
max_score, max_c, max_gamma = 0, 0, 0
c = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
sigma = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
gamma = [1/(2.0 * i * i) for i in sigma]
scores_mean = list()
for i in c:
    for j in gamma:
        clf_svm = SVC(C = i, gamma=j)
        scores = cross_val_score(clf_svm, tsne_X_train_20, y_train.values.ravel())
        score = scores.mean()
        scores_mean.append(score)
        if score > max_score:
            max_score = score
            max_c = i
            max_gamma = j
print("20")
print("max_c: ", max_c, " max_gamma: ", max_gamma, "max_score:", max_score)
print(scores_mean)
print("")
print("")


X_train_30 = pd.read_csv('X_train_rfe_30.csv', sep=',')
tsne = TSNE(n_components=2)
tsne_X_train_30 = tsne.fit_transform(X_train_30)

# RBF SVM
max_score, max_c, max_gamma = 0, 0, 0
c = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
sigma = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
gamma = [1/(2.0 * i * i) for i in sigma]
scores_mean = list()
for i in c:
    for j in gamma:
        clf_svm = SVC(C = i, gamma=j)
        scores = cross_val_score(clf_svm, tsne_X_train_30, y_train.values.ravel())
        score = scores.mean()
        scores_mean.append(score)
        if score > max_score:
            max_score = score
            max_c = i
            max_gamma = j
print("30")
print("max_c: ", max_c, " max_gamma: ", max_gamma, "max_score:", max_score)
print(scores_mean)
print("")
print("")

X_train_40 = pd.read_csv('X_train_rfe_40.csv', sep=',')
tsne = TSNE(n_components=2)
tsne_X_train_40 = tsne.fit_transform(X_train_40)

# RBF SVM
max_score, max_c, max_gamma = 0, 0, 0
c = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
sigma = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
gamma = [1/(2.0 * i * i) for i in sigma]
scores_mean = list()
for i in c:
    for j in gamma:
        clf_svm = SVC(C = i, gamma=j)
        scores = cross_val_score(clf_svm, tsne_X_train_40, y_train.values.ravel())
        score = scores.mean()
        scores_mean.append(score)
        if score > max_score:
            max_score = score
            max_c = i
            max_gamma = j
print("40")
print("max_c: ", max_c, " max_gamma: ", max_gamma, "max_score:", max_score)
print(scores_mean)
print("")
print("")