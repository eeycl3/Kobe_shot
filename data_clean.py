import pandas as pd
import csv
import numpy as np
pd.set_option('display.expand_frame_repr', False)
### open file read it as csv
csvFile = open("data.csv")
originalData = pd.read_csv(csvFile)
data = originalData
#data.set_index('shot_id', inplace=True)

""" drop data columns """
# game_event_id is independent
# game_id is related to game_date
# lat and lon are useless
# team_id and team_name are all the same
columns = ['game_event_id', 'game_id', 'lat', 'lon', 'team_id', 'team_name']
data.drop(columns, axis=1, inplace=True)

"""deal with categorical data"""
# deal with date.
data['game_date_DT'] = pd.to_datetime(data['game_date'])
data['dayOfWeek'] = data['game_date_DT'].dt.dayofweek
data['dayOfYear'] = data['game_date_DT'].dt.dayofyear
data['secondsFromPeriodEnd'] = 60*data['minutes_remaining']+data['seconds_remaining']
data['secondsFromPeriodStart'] = (data['period'] <= 4).astype(int)*(60*(11-data['minutes_remaining'])+(60-data['seconds_remaining'])) + (data['period'] > 4).astype(int)*(60*(4-data['minutes_remaining'])+(60-data['seconds_remaining']))
data['secondsFromGameStart'] = (data['period'] <= 4).astype(int)*((data['period']-1)*12*60) + data['secondsFromPeriodStart'] + (data['period'] > 4).astype(int)*((data['period']-5)*5*60 + 4*12*60)
data.drop("game_date", axis=1, inplace=True)
data.drop("minutes_remaining", axis=1, inplace=True)
data.drop("seconds_remaining", axis=1, inplace=True)
"""deal with matchup"""
list = []
for row in data["matchup"]:
    if "@" in str(row):
        list.append(0)
    elif "vs" in str(row):
        list.append(1)
data["home_field"] = pd.DataFrame(list)
data.drop("matchup", axis=1, inplace=True)

"""deal with shot_type"""
list2 = []
for row in data["shot_type"]:
    if "2PT" in str(row):
        list2.append(1)
    else:
        list2.append(0)
data["2PT"] = pd.DataFrame(list)
data.drop("shot_type", axis=1, inplace=True)

### spilt to different set.
##Header of the original csv
header = []
for col in data:
    header.append(col)
print (header)
### classify which one is predicting Set, which one is training Set
file1 = open("trainingSet.csv",'w',newline='')    # goal
file2 = open("predictSet.csv",'w',newline='') # data to predict
writer1 = csv.writer(file1)
writer2 = csv.writer(file2)
writer1.writerow(header)
writer2.writerow(header)
for index, row in data.iterrows():
    if row['shot_made_flag'] == 1 or row['shot_made_flag'] == 0:
        writer1.writerow(row)
    else:
        writer2.writerow(row)