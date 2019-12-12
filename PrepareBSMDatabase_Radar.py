# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 22:03:00 2019

@author: chen4416
"""

import sqlite3
from sqlite3 import Error
import csv
import numpy as np
import pandas as pd
import random as rd

date = "12_1"
penetration_rate = 1
save_to = "D:/Radar_dara_process_10_25/" + "radar_" + str(int(penetration_rate * 100)) + "_percent_" + date + ".csv"
read_in = "C:/Users/chen4416.AD/Dropbox/radar_dara_10_19/radar_trajectory_10_19.csv"

df_out = pd.DataFrame(
    columns=['Time', 'Vehicle_id', 'global_x', 'global_y', 'speed', 'latitude', 'longitude', 'distance', 'lane'])
df_in = pd.read_csv(read_in, sep=",")

vehicle_list = np.unique(df_in[["Vehicle_id"]])

for each_vehicle in vehicle_list:
    rand_number = rd.random()
    traj_veh = df_in[df_in.Vehicle_id == each_vehicle]
    if (rand_number < penetration_rate):
        print(1)
        df_out = df_out.append(traj_veh, ignore_index=True)
        # print(df_out)

df_out.to_csv(save_to, index=None, header=True)


