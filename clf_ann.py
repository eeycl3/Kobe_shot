from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
# fix random seed for reproducibility
np.random.seed(7)

X_train = pd.read_csv('X_train_rfe_20.csv', sep=',').as_matrix()
X_test = pd.read_csv('X_test_rfe_20.csv', sep=',').as_matrix()
y_train = pd.read_csv('y_train.csv', sep=',').as_matrix()
y_test = pd.read_csv('y_test.csv', sep=',').as_matrix()

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
max_score, max_i, max_j = 0, 0, 0
param = 20
matrix_mean_score = [[0 for i in range(param)] for i in range(param)]
for i in range(1,param):
    for j in range(1,param):
      cvscores = []
      for train, test in kfold.split(X_train, y_train):
          # create model
          model = Sequential()
          model.add(Dense(i, input_dim=20, activation='relu'))
          model.add(Dense(j, activation='relu'))
          model.add(Dense(1, activation='sigmoid'))
          # compile model
          model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
          # fit the model
          model.fit(X_train[train], y_train[train], epochs=50, batch_size=10)
          # evaluate the model
          scores = model.evaluate(X_train[test], y_train[test])
          cvscores.append(scores[1])
      matrix_mean_score[i][j] = np.mean(cvscores)
      if matrix_mean_score[i][j] > max_score:
          max_score = matrix_mean_score[i][j]
          max_i = i
          max_j = j
print("nodes num in 1st hidden layer: ", max_i, "nodes num in 2nd hidden layer: ", max_j, "score: ", max_score)
print(matrix_mean_score)
