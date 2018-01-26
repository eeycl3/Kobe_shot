import pandas
import csv
import scipy
import sys
from scipy.spatial import distance

pandas.set_option('display.expand_frame_repr', False)
### open file read it as csv
csvFile = open("data.csv")
originalData = pandas.read_csv(csvFile)

##Header of the original csv
header = []
for data in originalData:
    header.append(data)

### classify which one is predicting Set, which one is training Set
file1 = open("trainingSet.csv",'w',newline='')    # goal
file2 = open("predictSet.csv",'w',newline='') # data to predict
writer1 = csv.writer(file1)
writer2 = csv.writer(file2)
writer1.writerow(header)
writer2.writerow(header)
for index, row in originalData.iterrows():
    if row['shot_made_flag'] == 1 or row['shot_made_flag'] == 0:
        writer1.writerow(row)
    else:
        writer2.writerow(row)



