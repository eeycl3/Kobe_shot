import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


trainingSet = pd.read_csv('trainingSet.csv', sep=',')
y = trainingSet["shot_made_flag"]
dropColumn = ['game_date_DT', 'secondsFromPeriodStart', 'shot_made_flag', 'shot_id']
trainingSet.drop(dropColumn, axis=1, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(trainingSet, y, test_size=0.33, random_state=42)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

max_test_score = 0
for i in range(1, 200):
    for j in range(1, 200):
        clf_mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(i, j), random_state=0, activation='tanh')
        clf_mlp.fit(X_train, y_train)
        train_score = clf_mlp.score(X_train, y_train)
        test_score = clf_mlp.score(X_test, y_test)
        if test_score > max_test_score:
            related_train_score = train_score
            max_test_score = test_score
            max_i = i
            max_j = j
            best_clf_mlp = clf_mlp

print("2 hidden layer",max_i, max_j, "train score: ", related_train_score, "test score: ", max_test_score)

