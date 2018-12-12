# Preprocess data and create label for FER2013
# labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

import csv
import os
import numpy as np
import h5py
import warnings

warnings.filterwarnings('ignore')
file = 'data/fer2013.csv'

# Stage icon
stage = 0
# Creat the list to store the data and label information
Training_x = []
Training_y = []
PublicTest_x = []
PublicTest_y = []
PrivateTest_x = []
PrivateTest_y = []

datapath = os.path.join('data','data.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

with open(file,'r') as Data:
    data=csv.reader(Data)
    for row in data:
        # Training dataset
        if row[-1] == 'Training' and stage == 0:
            print('Stage 1, processing Training dataset')
            stage = 1

        if row[-1] == 'Training':
            temp_list = []
            for pixel in row[1].split( ):
                temp_list.append(int(pixel))
            I = np.asarray(temp_list)
            Training_y.append(int(row[0]))
            Training_x.append(I.tolist())

        # PublicTest dataset
        if row[-1] == 'PublicTest' and stage == 1:
            print('Stage 2, processing PublicTest data set')
            stage = 2

        if row[-1] == "PublicTest" :
            temp_list = []
            for pixel in row[1].split( ):
                temp_list.append(int(pixel))
            I = np.asarray(temp_list)
            PublicTest_y.append(int(row[0]))
            PublicTest_x.append(I.tolist())

        # PrivateTest dataset
        if row[-1] == 'PrivateTest' and stage == 2:
            print('Stage 2, processing PrivateTest data set')
            stage = 3
        if row[-1] == 'PrivateTest':
            temp_list = []
            for pixel in row[1].split( ):
                temp_list.append(int(pixel))
            I = np.asarray(temp_list)

            PrivateTest_y.append(int(row[0]))
            PrivateTest_x.append(I.tolist())

print(np.shape(Training_x))
print(np.shape(PublicTest_x))
print(np.shape(PrivateTest_x))

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("Training_data", dtype = 'uint8', data=Training_x)
datafile.create_dataset("Training_label", dtype = 'int64', data=Training_y)
datafile.create_dataset("PublicTest_data", dtype = 'uint8', data=PublicTest_x)
datafile.create_dataset("PublicTest_label", dtype = 'int64', data=PublicTest_y)
datafile.create_dataset("PrivateTest_data", dtype = 'uint8', data=PrivateTest_x)
datafile.create_dataset("PrivateTest_label", dtype = 'int64', data=PrivateTest_y)
datafile.close()

print("Preprocessing data finished!!!")
