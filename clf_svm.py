import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

trainingSet = pd.read_csv('trainingSet.csv', sep=',')
y = trainingSet["shot_made_flag"]

trainingSet.drop(['game_date_DT', 'secondsFromPeriodStart', 'shot_made_flag','shot_id'], axis=1, inplace=True)

# print(data)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(trainingSet)
data_norm = scaler.transform(trainingSet)

x_train, x_test, y_train, y_test = train_test_split(data_norm, y, test_size=0.3, random_state=42)

#svm = SVC()
#svm = SVC(kernel='linear')
svm = LinearSVC()
svm.fit(x_train, y_train)
score_svm_train = svm.score(x_train, y_train)
score_svm = svm.score(x_test, y_test)
print(score_svm_train)
print(score_svm)
