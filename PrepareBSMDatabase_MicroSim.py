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
penetration_rate = [0.05, 0.15, 0.25]
read_in = "D://BSM//create_partial_percentage_10_1//10_1_all_vehicles.csv"
df_in = pd.read_csv(read_in, sep=",")

for each_rate in penetration_rate:
    save_to = "D://BSM//create_partial_percentage_10_1//" + date + str(
        int(each_rate * 100)) + "_percent_" + "MicroSim_" + ".csv"
    df_out = pd.DataFrame()

    vehicle_list = np.unique(df_in[["vehicle_id"]])

    for each_vehicle in vehicle_list:
        rand_number = rd.random()
        traj_veh = df_in[df_in.vehicle_id == each_vehicle]
        if (rand_number < each_rate):
            # print(1)
            df_out = df_out.append(traj_veh, ignore_index=True)
            # print(df_out)
    df_out_1 = df_out.sort_values(by=['simulation_time'])
    df_out_1.to_csv(save_to, index=None, header=True)

date = "9_15"
penetration_rate = [0.05, 0.15, 0.25]
read_in = "D://BSM//all_vehicle_9_15_2019//9_15_all_vehicles.csv"
df_in = pd.read_csv(read_in, sep=",")

for each_rate in penetration_rate:
    save_to = "D://BSM//all_vehicle_9_15_2019//" + date + str(int(each_rate * 100)) + "_percent_" + "MicroSim_" + ".csv"
    df_out = pd.DataFrame()

    vehicle_list = np.unique(df_in[["vehicle_id"]])

    for each_vehicle in vehicle_list:
        rand_number = rd.random()
        traj_veh = df_in[df_in.vehicle_id == each_vehicle]
        if (rand_number < each_rate):
            # print(1)
            df_out = df_out.append(traj_veh, ignore_index=True)
            # print(df_out)
    df_out_2 = df_out.sort_values(by=['simulation_time'])
    df_out_2.to_csv(save_to, index=None, header=True)

