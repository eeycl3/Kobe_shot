from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE


X_train = pd.read_csv('X_train_rfe_20.csv', sep=',').as_matrix()
X_test = pd.read_csv('X_test_rfe_20.csv', sep=',').as_matrix()
y_train = pd.read_csv('y_train.csv', sep=',').as_matrix()
y_test = pd.read_csv('y_test.csv', sep=',').as_matrix()

tsne = TSNE(n_components=2)
tsne_X_train = tsne.fit_transform(X_train)

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
cvscores = []
for train, test in kfold.split(tsne_X_train, y_train):
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=tsne_X_train.shape[1], activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the model
    model.fit(tsne_X_train[train], y_train[train], epochs=50, batch_size=10)
    # evaluate the model
    scores = model.evaluate(tsne_X_train[test], y_train[test])
    cvscores.append(scores[1])