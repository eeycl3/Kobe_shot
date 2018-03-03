import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import csv

# training set preprocessing
trainingSet = pd.read_csv('trainingSet.csv', sep=',')
y = trainingSet["shot_made_flag"]
dropColumn = ['game_date_DT', 'secondsFromPeriodStart', 'shot_made_flag', 'shot_id']
trainingSet.drop(dropColumn, axis=1, inplace=True)

# data normalization
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(trainingSet)
data_norm = scaler.transform(trainingSet)

lda = LinearDiscriminantAnalysis()
data_lda_t = lda.fit(data_norm, y).transform(data_norm)

# predict set preprocessing
predictSet = pd.read_csv("predictSet.csv")
y_predictSet = predictSet['shot_made_flag']
p_id = predictSet['shot_id']

predictSet.drop(dropColumn, axis=1, inplace=True)

scaler_pre = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler_pre.fit(predictSet)
data_norm_pre = scaler_pre.transform(predictSet)

data_lda_p = lda.transform(data_norm_pre)

# lda_p = LinearDiscriminantAnalysis()
# data_lda_p = lda_p.fit(data_norm_pre, y_predictSet).transform(data_norm_pre)


# print(y)
# y_lda = pd.DataFrame([0, 1])

# x_train, x_test, y_train, y_test = train_test_split(data_lda, y, test_size=0.3, random_state=42)

svm = SVC(C=0.0023)
clf = CalibratedClassifierCV(svm)
clf.fit(data_lda_t, y)

svm_result = clf.predict_proba(data_lda_p)[:, 1]
svm_score = cross_val_score(clf, data_lda_t, y)


print("Accuracy: %0.2f (+/- %0.2f)" % (svm_score.mean(), svm_score.std() * 2))

file_svm = open("lda_svm_result.csv", 'w', newline='')
writer_svm = csv.writer(file_svm)
writer_svm.writerow(["shot_id", "shot_made_flag"])
for row in range(len(svm_result)):
    writer_svm.writerow([p_id[row], svm_result[row]])

# svm = SVC(kernel='linear')
# svm = LinearSVC()
# svm.fit(x_train, y_train)
# score_svm_train = svm.score(x_train, y_train)# jscore_svm = svm.score(x_test, y_test)
# print(score_svm_train)
# print(score_svm)

'''
file_lda_svm = open("lda_svm_result.csv", 'w', newline='')
writer1 = csv.writer(file_lda_svm)
writer1.writerow(["shot_id", "shot_made_flag"])
for index in range(len())
'''
