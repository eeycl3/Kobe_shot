from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


X_train = pd.read_csv('X_train_rfe_20.csv', sep=',').as_matrix()
X_test = pd.read_csv('X_test_rfe_20.csv', sep=',').as_matrix()
y_train = pd.read_csv('y_train.csv', sep=',').as_matrix()
y_test = pd.read_csv('y_test.csv', sep=',').as_matrix()

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
max_score, max_i, max_j = 0, 0, 0
num_1st = [2,5]
num_2nd = [2,5]
scorelist = []
for i in num_1st:
    for j in num_2nd:
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
      scorelist.append(np.mean(cvscores))
      if np.mean(cvscores) > max_score:
          max_score = np.mean(cvscores)
          max_i = i
          max_j = j
print("nodes num in 1st hidden layer: ", max_i, "nodes num in 2nd hidden layer: ", max_j, "score: ", max_score)
print(scorelist)
