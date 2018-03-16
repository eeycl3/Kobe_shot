import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

X_train = pd.read_csv('X_train_rfe_20.csv', sep=',')
y = pd.read_csv('y_train_rfe_20.csv', sep = ',')
y_train = y["shot_made_flag"]

max_score = 0
min_score_std = 999999999999999999999999
param = 50
matrix_mean_score = [[0] * param] * param
matrix_score_std = [[0] * param] * param
for i in range(1, param):
    for j in range(1, param):
        clf_mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(i, j), random_state=0, activation='tanh')
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

plt.figure()
im=plt.imshow(matrix_mean_score)
plt.title("MLP accuracy score with different units in 2 hidden layer")
plt.colorbar(im)
plt.xlim(1, param)
plt.ylim(1, param)
plt.xlabel("first hidden layer unit number")
plt.ylabel("second hidden layer unit number")
plt.savefig("MLP_param_figure.png")
print("2 hidden layer",max_i, max_j,"score: ", max_score, "std: ", min_score_std)
print(matrix_mean_score)
print(matrix_score_std)

