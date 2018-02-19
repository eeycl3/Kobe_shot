##### KOBE BRYANT SHOT EXPLORATION AND CLASSIFICATION


# IMPORT LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Arc
from sklearn import mixture
from sklearn import ensemble
from sklearn import model_selection
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import log_loss
import time
import itertools
import operator

from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier


# LOAD DATA

original_data = pd.read_csv('data.csv')


# DATA CLEANING

#game_event_id is independent
#game_id is related to game_date
#lat and lon are useless
#team_id and team_name are all the same

columns = ['game_event_id', 'game_id', 'lat', 'lon', 'team_id', 'team_name']
data = original_data.drop(columns, axis = 1)
#data.set_index('shot_id', inplace=True)

# Handling features one by one

# dealing with shot attepmted time 
data['game_date_DT'] = pd.to_datetime(data['game_date'])
data['dayOfWeek']    = data['game_date_DT'].dt.dayofweek
data['dayOfYear']    = data['game_date_DT'].dt.dayofyear
data.drop('game_date', axis = 1, inplace = True)

data['secondsFromPeriodEnd']   = 60*data['minutes_remaining']+data['seconds_remaining']
data['secondsFromPeriodStart'] = (data['period'] <= 4).astype(int)*(60*(11-data['minutes_remaining'])+(60-data['seconds_remaining'])) + (data['period'] > 4).astype(int)*(60*(4-data['minutes_remaining'])+(60-data['seconds_remaining']))
data['secondsFromGameStart']   = (data['period'] <= 4).astype(int)*((data['period']-1)*12*60) + data['secondsFromPeriodStart'] + (data['period'] > 4).astype(int)*((data['period']-5)*5*60 + 4*12*60)
data.drop('minutes_remaining', axis = 1, inplace = True)
data.drop('seconds_remaining', axis = 1, inplace = True)

# converting matchup to home and away game 
list = []
for row in data['matchup']:
    if '@' in str(row):
        list.append(0)
    else:
        list.append(1)
data['home_field'] = pd.DataFrame(list)
data.drop('matchup', axis = 1, inplace = True)

# feature shot_type handling 
list = []
for row in data['shot_type']:
    if str(row) == '2PT Field Goal':
        list.append(1)
    else:
        list.append(0)
data['2PT'] = pd.DataFrame(list)
data.drop('shot_type', axis = 1, inplace = True)

# feature handling one by one



#split data into training data and test data
#mask = (data['shot_made_flag'] == 0) | (data['shot_made_flag'] == 1)
#training_data = data[mask]
#test_data = data[~mask]








