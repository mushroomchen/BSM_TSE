# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 23:05:32 2019

@author: chen4416
"""
from os import listdir
from os.path import isfile, join, isdir
import pandas as pd

folder_auxliary = "D:\\radar_trajectory\\seperate_vehicle_trajectory\\Auxilary_vehicle1"
folder_left = "D:\\radar_trajectory\\seperate_vehicle_trajectory\\Left_vehicle1"
folder_middle = "D:\\radar_trajectory\\seperate_vehicle_trajectory\\Middle_vehicle1"
folder_right = "D:\\radar_trajectory\\seperate_vehicle_trajectory\\Right_vehicle1"

folder_list = [folder_left, folder_middle, folder_right]


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


df = pd.DataFrame(
    columns=['Time', 'Vehicle_id', 'global_x', 'global_y', 'speed', 'latitude', 'longitude', 'distance', 'lane'])

folder_counter = 0
for each_folder in folder_list:
    each_file_list = getListOfFiles(each_folder)
    lane = each_folder.split('\\')[3].split('_')[0]
    folder_counter = folder_counter + 1
    for each_file in each_file_list:
        print("file_size: ", len(df))
        veh_traj = pd.read_csv(each_file, sep=",")
        traj = veh_traj.drop(['Unnamed: 0'], axis=1)
        if (len(traj) > 0):
            vehicle_id = 1000 * folder_counter + int(each_file.split('\\')[4].split('.')[0].split('_')[1])
            for each_row in range(len(traj)):
                time = traj.iloc[each_row][1]
                x_cord = traj.iloc[each_row][2]
                y_cord = traj.iloc[each_row][3]
                speed = traj.iloc[each_row][4]
                Latitude = traj.iloc[each_row][5]
                Longitude = traj.iloc[each_row][6]
                distance = traj.iloc[each_row][7]
                df = df.append(
                    {'Time': time, 'Vehicle_id': vehicle_id, 'global_x': x_cord, 'global_y': y_cord, 'speed': speed,
                     'latitude': Latitude, 'longitude': Longitude, 'distance': distance, 'lane': lane},
                    ignore_index=True)

df.to_csv("D:\\radar_trajectory\\seperate_vehicle_trajectory\\radar_trajectory_10_19.csv", sep=',', mode='a',
          float_format='%.3f')