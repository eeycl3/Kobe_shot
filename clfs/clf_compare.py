import timeit
import pandas as pd
import numpy as np
from random import randint
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense


X_train_20 = pd.read_csv('X_train_rfe_20.csv', sep=',')
X_test_20 = pd.read_csv('X_test_rfe_20.csv', sep=',')
X_train_30 = pd.read_csv('X_train_rfe_30.csv', sep=',')
X_test_30 = pd.read_csv('X_test_rfe_30.csv', sep=',')
X_train_40 = pd.read_csv('X_train_rfe_40.csv', sep=',')
X_test_40 = pd.read_csv('X_test_rfe_40.csv', sep=',')

y_train = pd.read_csv('y_train.csv', sep=',')
y_test = pd.read_csv('y_test.csv', sep=',')
y_test = y_test["shot_made_flag"]
y_train = y_train["shot_made_flag"]

pca = PCA(n_components=2)
pca_X_train_20 = pca.fit_transform(X_train_20)
pca_X_test_20 = pca.transform(X_test_20)

train_time_svm_fea20=[]
train_time_svm_fea30=[]
train_time_svm_fea40=[]
train_time_rf_fea20=[]
train_time_rf_fea30=[]
train_time_rf_fea40=[]
train_time_svm_pca=[]
train_time_ann_hidden2=[]


classification_time_svm_fea20=[]
classification_time_svm_fea30=[]
classification_time_svm_fea40=[]
classification_time_rf_fea20=[]
classification_time_rf_fea30=[]
classification_time_rf_fea40=[]
classification_time_svm_pca=[]
classification_time_ann_hidden2=[]

acu_svm_fea20=[]
acu_svm_fea30=[]
acu_svm_fea40=[]
acu_rf_fea20=[]
acu_rf_fea30=[]
acu_rf_fea40=[]
acu_svm_pca=[]
acu_ann_hidden2=[]

pre_svm_fea20=[]
pre_svm_fea30=[]
pre_svm_fea40=[]
pre_rf_fea20=[]
pre_rf_fea30=[]
pre_rf_fea40=[]
pre_svm_pca=[]
pre_ann_hidden2=[]

rec_svm_fea20=[]
rec_svm_fea30=[]
rec_svm_fea40=[]
rec_rf_fea20=[]
rec_rf_fea30=[]
rec_rf_fea40=[]
rec_svm_pca=[]
rec_ann_hidden2=[]

f1_svm_fea20=[]
f1_svm_fea30=[]
f1_svm_fea40=[]
f1_rf_fea20=[]
f1_rf_fea30=[]
f1_rf_fea40=[]
f1_svm_pca=[]
f1_ann_hidden2=[]

# ann feas20 hidden2

# ann model creation
model = Sequential()
model.add(Dense(10, input_dim=20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

for i in range(3):
    # fit the model
    start = timeit.default_timer()
    model.fit(X_train_20, y_train, epochs=50, batch_size=10)
    stop = timeit.default_timer()
    train_time_ann_hidden2.append(stop - start)
    # evaluate the model
    start = timeit.default_timer()
    predictions = model.predict(X_test_20)
    stop = timeit.default_timer()
    classification_time_ann_hidden2.append(stop - start)
    # round predictions
    ann_pred = [round(x[0]) for x in predictions]

    score = accuracy_score(y_test, ann_pred)
    acu_ann_hidden2.append(score)

    pre = precision_score(y_test, ann_pred)
    pre_ann_hidden2.append(pre)

    rec = recall_score(y_test, ann_pred)
    rec_ann_hidden2.append(rec)

    f1 = f1_score(y_test, ann_pred)
    f1_ann_hidden2.append(f1)

for i in range(20):
    
    #svm_fea20
    svm_fea20 = clf_svm = SVC(C = 10, gamma=0.02)
    start = timeit.default_timer()
    svm_fea20.fit(X_train_20, y_train)
    stop = timeit.default_timer()
    train_time_svm_fea20.append(stop - start)

    start = timeit.default_timer()
    svm_pred = svm_fea20.predict(X_test_20)
    stop = timeit.default_timer()
    classification_time_svm_fea20.append(stop - start)

    score = accuracy_score(y_test, svm_pred)
    acu_svm_fea20.append(score)

    pre = precision_score(y_test, svm_pred)
    pre_svm_fea20.append(pre)

    rec = recall_score(y_test, svm_pred)
    rec_svm_fea20.append(rec)

    f1 = f1_score(y_test, svm_pred)
    f1_svm_fea20.append(f1)

    # svm_fea30
    svm_fea30 = SVC(C=2, gamma=0.02)
    start = timeit.default_timer()
    svm_fea30.fit(X_train_30, y_train)
    stop = timeit.default_timer()
    train_time_svm_fea30.append(stop - start)

    start = timeit.default_timer()
    svm_pred = svm_fea30.predict(X_test_30)
    stop = timeit.default_timer()
    classification_time_svm_fea30.append(stop - start)

    score = accuracy_score(y_test, svm_pred)
    acu_svm_fea30.append(score)

    pre = precision_score(y_test, svm_pred)
    pre_svm_fea30.append(pre)

    rec = recall_score(y_test, svm_pred)
    rec_svm_fea30.append(rec)

    f1 = f1_score(y_test, svm_pred)
    f1_svm_fea30.append(f1)

    # svm_fea40
    svm_fea40 = SVC(C=2, gamma=0.02)
    start = timeit.default_timer()
    svm_fea40.fit(X_train_40, y_train)
    stop = timeit.default_timer()
    train_time_svm_fea40.append(stop - start)

    start = timeit.default_timer()
    svm_pred = svm_fea40.predict(X_test_40)
    stop = timeit.default_timer()
    classification_time_svm_fea40.append(stop - start)

    score = accuracy_score(y_test, svm_pred)
    acu_svm_fea40.append(score)

    pre = precision_score(y_test, svm_pred)
    pre_svm_fea40.append(pre)

    rec = recall_score(y_test, svm_pred)
    rec_svm_fea40.append(rec)

    f1 = f1_score(y_test, svm_pred)
    f1_svm_fea40.append(f1)


    #rf_fea20
    rf_fea20 = RandomForestClassifier(random_state=0, n_estimators=17, max_depth=8)
    start = timeit.default_timer()
    rf_fea20.fit(X_train_20, y_train)
    stop = timeit.default_timer()
    train_time_rf_fea20.append(stop - start)

    start = timeit.default_timer()
    rf_pred = rf_fea20.predict(X_test_20)
    stop = timeit.default_timer()
    classification_time_rf_fea20.append(stop - start)

    score = accuracy_score(y_test, rf_pred)
    acu_rf_fea20.append(score)

    pre = precision_score(y_test, rf_pred)
    pre_rf_fea20.append(pre)

    rec = recall_score(y_test, rf_pred)
    rec_rf_fea20.append(rec)

    f1 = f1_score(y_test, rf_pred)
    f1_rf_fea20.append(f1)

    #rf_fea30
    rf_fea30 = RandomForestClassifier(random_state=0, n_estimators=25, max_depth=7)
    start = timeit.default_timer()
    rf_fea30.fit(X_train_30, y_train)
    stop = timeit.default_timer()
    train_time_rf_fea30.append(stop - start)

    start = timeit.default_timer()
    rf_pred = rf_fea30.predict(X_test_30)
    stop = timeit.default_timer()
    classification_time_rf_fea30.append(stop - start)

    score = accuracy_score(y_test, rf_pred)
    acu_rf_fea30.append(score)

    pre = precision_score(y_test, rf_pred)
    pre_rf_fea30.append(pre)

    rec = recall_score(y_test, rf_pred)
    rec_rf_fea30.append(rec)

    f1 = f1_score(y_test, rf_pred)
    f1_rf_fea30.append(f1)

    # rf_fea40
    rf_fea40 = RandomForestClassifier(random_state=0, n_estimators=32, max_depth=10)
    start = timeit.default_timer()
    rf_fea40.fit(X_train_40, y_train)
    stop = timeit.default_timer()
    train_time_rf_fea40.append(stop - start)

    start = timeit.default_timer()
    rf_pred = rf_fea40.predict(X_test_40)
    stop = timeit.default_timer()
    classification_time_rf_fea40.append(stop - start)

    score = accuracy_score(y_test, rf_pred)
    acu_rf_fea40.append(score)

    pre = precision_score(y_test, rf_pred)
    pre_rf_fea40.append(pre)

    rec = recall_score(y_test, rf_pred)
    rec_rf_fea40.append(rec)

    f1 = f1_score(y_test, rf_pred)
    f1_rf_fea40.append(f1)

    # svm_fea_pca 20
    svm_pca = SVC(C=0.5, gamma=200)
    start = timeit.default_timer()
    svm_pca.fit(pca_X_train_20, y_train)
    stop = timeit.default_timer()
    train_time_svm_pca.append(stop - start)

    start = timeit.default_timer()
    svm_pred = svm_pca.predict(pca_X_test_20)
    stop = timeit.default_timer()
    classification_time_svm_pca.append(stop - start)

    score = accuracy_score(y_test, svm_pred)
    acu_svm_pca.append(score)

    pre = precision_score(y_test, svm_pred)
    pre_svm_pca.append(pre)

    rec = recall_score(y_test, svm_pred)
    rec_svm_pca.append(rec)

    f1 = f1_score(y_test, svm_pred)
    f1_svm_pca.append(f1)

value = [[np.mean(train_time_svm_fea20), np.mean(train_time_svm_fea30), np.mean(train_time_svm_fea40), np.mean(train_time_rf_fea20), np.mean(train_time_rf_fea30), np.mean(train_time_rf_fea40), np.mean(train_time_svm_pca), np.mean(train_time_ann_hidden2)],
         [np.mean(classification_time_svm_fea20), np.mean(classification_time_svm_fea30),
          np.mean(classification_time_svm_fea40), np.mean(classification_time_rf_fea20),
          np.mean(classification_time_rf_fea30), np.mean(classification_time_rf_fea40),
          np.mean(classification_time_svm_pca),
          np.mean(classification_time_ann_hidden2)],
         [np.mean(acu_svm_fea20), np.mean(acu_svm_fea30), np.mean(acu_svm_fea40), np.mean(acu_rf_fea20),
          np.mean(acu_rf_fea30), np.mean(acu_rf_fea40), np.mean(acu_svm_pca), np.mean(acu_ann_hidden2)],
         [np.mean(pre_svm_fea20), np.mean(pre_svm_fea30), np.mean(pre_svm_fea40), np.mean(pre_rf_fea20),
          np.mean(pre_rf_fea30), np.mean(pre_rf_fea40), np.mean(pre_svm_pca), np.mean(pre_ann_hidden2)],
         [np.mean(rec_svm_fea20), np.mean(rec_svm_fea30), np.mean(rec_svm_fea40), np.mean(rec_rf_fea20),
          np.mean(rec_rf_fea30), np.mean(rec_rf_fea40), np.mean(rec_svm_pca), np.mean(rec_ann_hidden2)],
         [np.mean(f1_svm_fea20), np.mean(f1_svm_fea30), np.mean(f1_svm_fea40), np.mean(f1_rf_fea20),
          np.mean(f1_rf_fea30), np.mean(f1_rf_fea40), np.mean(f1_svm_pca), np.mean(f1_ann_hidden2)]
        ]
print(value)

