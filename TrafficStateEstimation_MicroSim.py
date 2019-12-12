# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:27:40 2019

@author: chen4416

"""
"""
there are some problems with the update of the covariance matrix

"""

import sys

sys.path.append('<directory path of Import_BSM_AIMSUN.py>')
import numpy as np
import math
import pandas as pd
import sqlite3
import csv
from sqlite3 import Error
import matplotlib.pyplot as plt
import seaborn as sns

global start_time
start_time = 300
global end_time
end_time = 3000
global time_step
time_step = 6
global link_length
# link_length = 4200.5 ###  4649 feet
link_length = 4649
today_date = "12_12"
penetration = "_15_percent"
actual_state_address = "D://BSM//all_vehicle_9_15_2019//actual_traffic_state_9_15.csv"


def main():
    penetration_rate = 0.6
    address = "D://BSM//all_vehicle_9_15_2019//9_15"+penetration+"_vehicles.csv"
    db_file = "D://BSM_CODE_TEST_12_12_2019//BSM_12_12_15_percent.sqlite"
    #save_to = "D://BSM//all_vehicle_9_15_2019//Sim_uncongested" + penetration + "_" + today_date
    save_to = "D://BSM_CODE_TEST_12_12_2019"
    global counter
    counter = 1
    global cell_length
    # cell_length = pe.initialize_ffs()*time_step*5280.0/3600     #in feet
    cell_length = 68.5 * time_step * 5280.0 / 3600
    global cell_number
    cell_number = int(math.ceil(link_length / cell_length))  # in feet
    # all_data = pd.read_csv(address)  use pandas package
    # list(all_data.columns.values)
    boundaries = calculate_bound_locations(link_length, cell_length, cell_number)
    link_lane_number = 3
    cell_length_in_mile = cell_length / 5280.0
    ####### initialize the Kalman Filter #######
    ####### set up matrices to store data #######

    total_time_steps = int((end_time - start_time) / time_step)
    KF_occ = np.zeros(shape=(total_time_steps, cell_number))  ### the estimated occupancy with KF
    mea_occ = np.zeros(shape=(total_time_steps, cell_number))  ### the measured occupancy
    mea_occ_via_speed = np.zeros(
        shape=(total_time_steps, cell_number))  ### the occupancy calculated with measured speeds
    pre_occ = np.zeros(shape=(total_time_steps, cell_number))  ### the prediceted occupacny with CTM
    KF_flow = np.zeros(shape=(total_time_steps, cell_number + 1))  ### the estimated flow with KF
    mea_flow = np.zeros(shape=(total_time_steps, cell_number + 1))  ### measured flow  ???? don't know how to get it ???
    measurement_speed = np.zeros(shape=(total_time_steps, cell_number))  ### measured speed
    KF_speed = np.zeros(
        shape=(total_time_steps, cell_number))  ### speeds calculated using the estimated occupancy values
    pre_speed = np.zeros(shape=(total_time_steps, cell_number))  ### the predicted speed

    Cell_travel_times = np.zeros(shape=(total_time_steps, cell_number))
    Link_travel_times = np.zeros(shape=(total_time_steps, 1))
    Link_queue_length = np.zeros(shape=(total_time_steps, 1))

    KF_capacity = np.zeros(shape=(total_time_steps, 1))  ### capacity
    KF_capacity[0, 0] = 2300 * link_lane_number  ### initialized capacity
    KF_ffs = np.zeros(shape=(total_time_steps, 1))  ### free-flow speed
    KF_ffs[0, 0] = 60  ### initilized speed
    KF_bws = np.zeros(shape=(total_time_steps, 1))  ### bws
    KF_bws[0, 0] = 16
    # average_car_length = pe.get_average_car_length()
    average_car_length = 15
    jam_density = int(5280 / (average_car_length * 1.5)) * link_lane_number
    InputFlowRate = 0
    KF_occ_1_min = np.zeros(shape=(int((end_time - start_time) * 1.0 / 60.0), cell_number))
    KF_flow_1_min = np.zeros(shape=(int((end_time - start_time) * 1.0 / 60.0), cell_number))
    KF_speed_1_min = np.zeros(shape=(int((end_time - start_time) * 1.0 / 60.0), cell_number))

    ####### parameters for the kalman filter #######  P,Q,R should be in the form of matrices
    Q_occ = 0.2
    Q_cap = 0.1
    Q_ffs = 0.1
    Q_sws = 0.1
    Q_speed = 0.2
    Q_input = 0.1
    P_cap = 0.8
    P_ffs = 0.8
    P_sws = 0.8
    P_input = 0.9
    R_occ = 0.75
    R_cap = 0.1
    R_ffs = 0.1
    R_sws = 0.1
    R_input = 0.7
    R_speed = 0.7
    KG_cap = 0.5
    KG_ffs = 0.5
    KG_sws = 0.5
    P_occ = np.zeros(shape=(total_time_steps, cell_number))
    P_speed = np.zeros(shape=(total_time_steps, cell_number))
    P_occ[0, :] = [0.9] * cell_number
    P_speed[0, :] = [0.2] * cell_number

    try:
        with open(address, 'r') as data_csv:
            raw_data = list(csv.reader(data_csv, delimiter=','))
            timelist = np.arange(start_time, end_time, 0.1)
            for t in timelist:
                #######################################################################
                ###################### every 0.1 second ###############################
                #######################################################################
                with sqlite3.connect(db_file) as conn:
                    data_to_submit = []
                    cur = conn.cursor()
                    cur.execute("PRAGMA synchronous = OFF")
                    cur.execute("BEGIN TRANSACTION")
                    for eachLine in raw_data[counter:]:
                        simulation_time = float(eachLine[1])
                        if (round(simulation_time, 1) == round(t, 1)):
                            counter = counter + 1  ## it is not the problem of sqlite
                            vehicle_id = int(eachLine[0])
                            simulation_time = round(float(eachLine[1]), 1)
                            section_id = int(eachLine[2])
                            segment_id = int(eachLine[3])
                            lane_number = int(eachLine[4])
                            current_pos_section = round(float(eachLine[5]), 3)
                            distance_end_section = round(float(eachLine[6]), 3)
                            world_pos_x = round(float(eachLine[7]), 3)
                            world_pos_y = round(float(eachLine[8]), 3)
                            world_pos_z = round(float(eachLine[9]), 3)
                            world_pos_x_rear = round(float(eachLine[10]), 3)
                            world_pos_y_rear = round(float(eachLine[11]), 3)
                            world_pos_z_rear = round(float(eachLine[12]), 3)
                            current_speed = float(eachLine[13])
                            distance_traveled = float(eachLine[14])  # in the network
                            section_entrance_time = float(eachLine[15])
                            current_stop_time = float(eachLine[16])
                            speed_drop = 0
                            # speed_drop = pe.get_speed_drop(conn, simulation_time, vehicle_id, current_speed)
                            time_step_id = int(math.ceil((simulation_time - start_time) / time_step))
                            cell_id = int(calculate_cell_id(abs(current_pos_section)))
                            data = (
                            vehicle_id, simulation_time, section_id, segment_id, lane_number, current_pos_section,
                            distance_end_section, \
                            world_pos_x, world_pos_y, world_pos_z, world_pos_x_rear, world_pos_y_rear, world_pos_z_rear,
                            current_speed, \
                            distance_traveled, section_entrance_time, current_stop_time, speed_drop, time_step_id,
                            cell_id)

                            data_to_submit.append(data)
                        else:
                            break
                    try:
                        cur.executemany(''' INSERT INTO BSM(vehicle_id,simulation_time,section_id,segment_id,lane_number,
                        current_pos_section, distance_end_section,world_pos_x,world_pos_y,world_pos_z,world_pos_x_rear,world_pos_y_rear,
                        world_pos_z_rear,current_speed,distance_traveled,section_entrance_time,current_stop_time,speed_drop,time_step_id,cell_id)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', data_to_submit)

                    except Error as e:
                        print(e)
                    conn.commit()

                    ##### adjust probe traffic state table #####
                    ########start to apply Kalman filter#######
                    if ((round(t, 1) % time_step) == 0):
                        ##### get estimation from the function
                        ##### get input flow #####
                        if (round(t, 1) == start_time):
                            [mean_speeds, probe_traffic_states, input_flow] = Time_space_data_extraction(conn,
                                                                                                         round(t, 1),
                                                                                                         start_time,
                                                                                                         boundaries,
                                                                                                         cell_length)

                            occ_measurement = probe_traffic_states[0][1:] * cell_length / 5280
                            flow_measurement = probe_traffic_states[1][:] * time_step / 3600
                            measurement_speed[0, :] = mean_speeds
                            mea_occ[0][:] = occ_measurement
                            mea_flow[0][:] = flow_measurement
                            mea_occ_via_speed[0][:] = occ_measurement

                            InputFlowRate = int(input_flow) / penetration_rate

                            KF_occ[0, :] = mea_occ[0][:]
                            KF_flow[0, :] = mea_flow[0][:]
                            P_occ[1, :] = P_occ[0, :]

                        else:
                            i = int((round(t, 1) - start_time) * 1.0 / time_step)
                            print("Predict time step: ", i)
                            [mean_speeds, probe_traffic_states, input_flow] = Time_space_data_extraction(conn,
                                                                                                         round(t, 1),
                                                                                                         start_time,
                                                                                                         boundaries,
                                                                                                         cell_length)
                            occ_measurement = probe_traffic_states[0][1:] * cell_length / 5280
                            flow_measurement = probe_traffic_states[1][:] * time_step / 3600

                            mea_occ[i][:] = occ_measurement
                            mea_flow[i][:] = flow_measurement

                            # InputFlowRate_pre = InputFlowRate
                            # InputFlowRate_mea = int(input_flow)/penetration_rate
                            # InputFlowRate_mea = 1/penetration_rate
                            # KG_input = P_input/(P_input+R_input)
                            # P_input = max(0, P_input - KG_input*P_input)
                            # InputFlowRate = InputFlowRate_pre + KG_input*(InputFlowRate_mea-InputFlowRate_pre)

                            InputFlowRate = input_flow / penetration_rate
                            # InputFlowRate = int(actual_inflow.iloc[i])
                            # print("Time",i,"InputFlowRate: ",InputFlowRate)
                            print("input flow: ", InputFlowRate)

                            # KF_capacity[i][0]= 6800.0
                            KF_bws[i][0] = 14.34
                            KF_ffs[i][0] = 60.9
                            KF_capacity[i][0] = 2200 * link_lane_number
                            # KF_capacity[i][0] = jam_density/(1.0/(KF_bws[i][0])+1.0/(KF_ffs[i][0]))

                            for j in range(cell_number + 1):
                                if (j == 0):
                                    KF_flow[i][j] = InputFlowRate

                                elif (j == cell_number):
                                    KF_flow[i][j] = min(KF_occ[i - 1][j - 1], KF_capacity[i][0] * time_step / 3600,
                                                        (KF_bws[i][0] / KF_ffs[i][0] * 1.0) * (
                                                                    jam_density * cell_length_in_mile - KF_occ[i - 1][
                                                                j - 1]))
                                    # KF_flow[i][j] = KF_flow[i][j]
                                    # KF_flow[i][j] = min((mea_occ[i-1][j-1]*KF_ffs[i][0]/penetration_rate), KF_capacity[i][0]*time_step/3600, jam_density*KF_bws[i][0]*(cell_length*1.0/5280)/(mean_speeds[j-1]+KF_bws[i][0])*mean_speeds[j-1])

                                else:
                                    Capacity1 = KF_capacity[i][0]
                                    ffspeed1 = KF_ffs[i][0]
                                    swspeed1 = KF_bws[i][0]
                                    KF_flow[i][j] = min(KF_occ[i - 1][j - 1], Capacity1 * time_step / 3600,
                                                        (swspeed1 / ffspeed1 * 1.0) * (
                                                                    jam_density * cell_length / 5280 - KF_occ[i - 1][
                                                                j]))
                                    # KF_flow[i][j] = KF_flow[i][j]
                                    # KF_flow[i][j] = min((mea_occ[i-1][j-1]*KF_ffs[i][0]/penetration_rate), KF_capacity[i][0]*time_step/3600, jam_density*KF_bws[i][0]*(cell_length*1.0/5280)/(mean_speeds[j-1]+KF_bws[i][0])*mean_speeds[j-1])

                            print("mean_speeds: ", mean_speeds)
                            for j in range(cell_number):  ##the ith cell has j = i-1
                                pre_occ[i][j] = KF_occ[i - 1][j] + KF_flow[i][j] - KF_flow[i][j + 1] + rand(Q_occ)
                                if (pre_occ[i][j] < 0):
                                    print ("Measurement errors: negative occupancy ")
                                    pre_occ[i][j] = max(0, pre_occ[i][j])

                                if (mean_speeds[j] > 0):
                                    if (mean_speeds[j] >= KF_ffs[i][0] * 0.9):
                                        # print("occ measurement type 1")
                                        # critical_occ = KF_capacity[i,0]/KF_ffs[i][0]
                                        # Occ_mea = 0.5*critical_occ*(cell_length/5280)
                                        # elif (mean_speeds[j] > (KF_capacity[i,0]/(jam_density-(KF_capacity[i,0]/KF_bws[i,0])))):
                                        Occ_mea = mea_occ[i][j] / penetration_rate
                                    # print("occ measurement type 2")
                                    # Occ_mea = KF_capacity[i,0]*(cell_length/5280)/mean_speeds[j]
                                    else:
                                        # print("occ measurement type 2")
                                        # Occ_mea = jam_density*(cell_length/5280)/(1+(KF_ffs[i][0]/KF_bws[i][0]))
                                        Occ_mea = min(KF_capacity[i][0] * cell_length_in_mile / mean_speeds[j],
                                                      jam_density * KF_bws[i][0] * (cell_length * 1.0 / 5280) / (
                                                                  mean_speeds[j] + KF_bws[i][0]))
                                else:
                                    # print("occ measurement type 4")
                                    Occ_mea = max(0, KF_occ[i - 1][j] + abs(rand(R_occ)))
                                mea_occ_via_speed[i][j] = Occ_mea

                                P_occ[i][j] = P_occ[i - 1][j] + Q_occ
                                if (P_occ[i][j] < 0):
                                    print("##########################ERROR################################")
                                ##### the equation for Kalman Gain needs to be modified.
                                ##### should includes H function corresponding to the measurement

                                KalmanGain = P_occ[i][j] * 1.0 / (P_occ[i][j] + R_occ)
                                KF_occ[i][j] = max(0, pre_occ[i][j] + KalmanGain * (
                                            mea_occ_via_speed[i][j] - pre_occ[i][j]))
                                P_occ[i][j] = P_occ[i][j] - KalmanGain * P_occ[i][j]

                                ##### calculate KF_speed ####
                                Density = KF_occ[i][j] / cell_length_in_mile
                                option1 = KF_ffs[i, 0]
                                option2 = KF_capacity[i, 0] / Density
                                option3 = (jam_density - Density) * KF_bws[i, 0] / Density
                                pre_speed[i][j] = min(option1, option2, option3) + rand(Q_speed)
                                mea_speed = mean_speeds[j]
                                P_speed[i][j] = P_speed[i][j] + Q_speed
                                KG_speed = P_speed[i][j] / (P_speed[i][j] + R_speed)
                                KF_speed[i][j] = (1 - KG_speed) * pre_speed[i][j] + KG_speed * mea_speed
                                P_speed[i][j] = P_speed[i][j] - KG_speed * P_speed[i][j]

                                if (mean_speeds[j] > 0):
                                    measurement_speed[i, j] = mean_speeds[j]
                                    Cell_travel_times[i, j] = (boundaries[j + 1] - boundaries[j]) / (
                                                mean_speeds[j] * 1.46667)

                                else:
                                    measurement_speed[i, j] = KF_speed[i][j]
                                    Cell_travel_times[i, j] = (boundaries[j + 1] - boundaries[j]) / (
                                                KF_speed[i][j] * 1.46667)

                            Link_travel_times[i, 0] = sum(Cell_travel_times[i, :])

                            # print("Time Step: ", i, " Cell: ", j+1, " Speed: ", KG_speed[i][j])

                        # fig = plt.figure(figsize=(10, 3.5), dpi=100)
                        # plt.plot(KF_occ[:, 2], linewidth=0.8)
                        # plt.plot(mea_occ[:, 2] * (1.0), linewidth=0.3)
                        # plt.plot(mea_occ_via_speed[:, 2], linewidth=0.3)
                        # plt.plot(pre_occ[:, 2], linewidth=0.3)
                        # plt.xlabel('time step')
                        # plt.ylabel('occupancy (veh)')
                        # plt.legend(
                        #     ['Estimation', 'Measured Occ via Occ', 'Measured Occ via Speed', 'Predict Occ via CTM'])
                        # plt.show(fig)
            print("all row number: ", counter)

            ################################ finish adding new raw data in the database ################################
            ########## to different 0.1 time frame
            for j in range(cell_number):
                for i in range(total_time_steps):
                    if ((i + 1) % 10 == 0):
                        time_ID = (i + 1) / 10
                        occ_1_min = sum(KF_occ[i - 9:i + 1, j]) * 1.0 / (cell_length / 5280.0)
                        flow_1_min = sum(KF_flow[i - 9:i + 1, j]) * 60
                        speed_1_min = sum(KF_speed[i - 9:i + 1, j]) / (60 / time_step)

                        KF_occ_1_min[time_ID - 1][j] = occ_1_min
                        KF_flow_1_min[time_ID - 1][j] = flow_1_min
                        KF_speed_1_min[time_ID - 1][j] = speed_1_min

            try:
                actual_traffic_state = pd.read_csv(actual_state_address)

                actual_occ_matrix = np.zeros(shape=(450, cell_number))
                actual_flow_matrix = np.zeros(shape=(450, cell_number))
                actual_speed_matrix = np.zeros(shape=(450, cell_number))

                for row_index in range(len(actual_traffic_state)):
                    row_id = row_index / cell_number
                    col_id = row_index % cell_number
                    actual_occ_matrix[row_id, col_id] = actual_traffic_state.iloc[row_index, 3] * (
                                (boundaries[col_id + 1] - boundaries[col_id]) / 5280)
                    actual_flow_matrix[row_id, col_id] = actual_traffic_state.iloc[row_index, 2] * time_step / 3600.0
                    actual_speed_matrix[row_id, col_id] = actual_traffic_state.iloc[row_index, 4]

                occ_file = save_to + "\\KF_occ_" + today_date + penetration + ".csv"
                np.savetxt(occ_file, KF_occ, delimiter=',')
                occ_file_mea_speed = save_to + "\\KF_occ_mea_via_speed_" + today_date + penetration + ".csv"
                np.savetxt(occ_file_mea_speed, mea_occ_via_speed, delimiter=',')
                occ_file_predict_CTM = save_to + "\\KF_occ_pre_via_CTM_" + today_date + penetration + ".csv"
                np.savetxt(occ_file_predict_CTM, pre_occ, delimiter=',')
                flow_file = save_to + "\\KF_flow_" + today_date + penetration + ".csv"
                np.savetxt(flow_file, KF_flow, delimiter=',')
                speed_file = save_to + "\\pre_speed_" + today_date + penetration + ".csv"
                np.savetxt(speed_file, pre_speed, delimiter=',')
                speed_file = save_to + "\\KF_speed_" + today_date + penetration + ".csv"
                np.savetxt(speed_file, KF_speed, delimiter=',')

                KF_occ_1_min_file = save_to + "\\KF_occ_1_min_" + today_date + penetration + ".csv"
                np.savetxt(KF_occ_1_min_file, KF_occ_1_min, delimiter=',')
                KF_flow_1_min_file = save_to + "\\KF_flow_1_min_" + today_date + penetration + ".csv"
                np.savetxt(KF_flow_1_min_file, KF_flow_1_min, delimiter=',')
                KF_speed_1_min_file = save_to + "\\KF_speed_1_min_" + today_date + penetration + ".csv"
                np.savetxt(KF_speed_1_min_file, KF_speed_1_min, delimiter=',')

                parameter_file = save_to + "\\parameters_" + today_date + penetration + ".csv"
                parameters = np.zeros(shape=(total_time_steps, 3))
                parameters[:, 0] = KF_capacity[:, 0]
                parameters[:, 1] = KF_ffs[:, 0]
                parameters[:, 2] = KF_bws[:, 0]
                np.savetxt(parameter_file, parameters, delimiter=',')

                numerator_occ = np.zeros(shape=(1, cell_number))
                denominator_occ = np.zeros(shape=(1, cell_number))
                total_estimation_error_occ = 0
                each_cell_error_occ = np.zeros(shape=(1, cell_number))

                numerator_speed = np.zeros(shape=(1, cell_number))
                denominator_speed = np.zeros(shape=(1, cell_number))
                total_estimation_error_speed = 0
                each_cell_error_speed = np.zeros(shape=(1, cell_number))

                numerator_flow = np.zeros(shape=(1, cell_number))
                denominator_flow = np.zeros(shape=(1, cell_number))
                total_estimation_error_flow = 0
                each_cell_error_flow = np.zeros(shape=(1, cell_number))

                for pl in range(cell_number):
                    actual_cell_density = list(
                        actual_traffic_state.loc[actual_traffic_state.iloc[:, 1] == pl + 1].iloc[:, 3])
                    actual_cell_occupancy = []
                    for each_density in actual_cell_density:
                        actual_cell_occupancy.append(each_density * ((boundaries[pl + 1] - boundaries[pl]) / 5280))

                    actual_flow = list(actual_traffic_state.loc[actual_traffic_state.iloc[:, 1] == pl + 1].iloc[:, 2])
                    actual_cell_flow = []
                    for each_flow in actual_flow:
                        actual_cell_flow.append(each_flow / 600.0)

                    fig = plt.figure(figsize=(6, 3.5), dpi=100)
                    print("################ Cell", pl + 1, "##################")
                    plt.plot(KF_occ[:, pl], linewidth=0.8)
                    plt.plot(mea_occ[:, pl] * (1.0 / penetration_rate), linewidth=0.6)
                    plt.plot(actual_cell_occupancy, linewidth=0.6)
                    plt.xlabel('time step')
                    plt.ylabel('occupancy (veh)')
                    plt.legend(['Estimated Occupancy', 'Measured Occupancy', 'Actual Density'])
                    plt.show(fig)

                    actual_cell_speed = list(
                        actual_traffic_state.loc[actual_traffic_state.iloc[:, 1] == pl + 1].iloc[:, 4])
                    fig = plt.figure(figsize=(6, 3.5), dpi=100)
                    plt.plot(measurement_speed[:, pl], linewidth=0.8)
                    plt.plot(actual_cell_speed, linewidth=0.6)
                    plt.xlabel('time step')
                    plt.ylabel('speed (mph)')
                    plt.legend(['KF_speed', 'actual_cell_speed'])
                    plt.show(fig)

                    fig = plt.figure(figsize=(6, 3.5), dpi=100)
                    plt.plot(KF_flow[:, pl], linewidth=0.8)
                    plt.plot(actual_cell_flow, linewidth=0.6)
                    plt.xlabel('time step')
                    plt.ylabel('flow (vph)')
                    plt.legend(['KF_flow', 'actual_cell_flow'])
                    plt.show(fig)

                    estimated_cell_density = KF_occ[:, pl]
                    estimated_cell_speed = measurement_speed[:, pl]
                    estimated_cell_flow = KF_flow[:, pl]

                    if (len(actual_cell_density) != len(estimated_cell_density)):
                        print("ERROR: unmatched list length", len(actual_cell_density), len(estimated_cell_density))

                    else:
                        for each_element in range(len(actual_cell_density)):
                            numerator_occ[0, pl] = numerator_occ[0, pl] + abs(
                                estimated_cell_density[each_element] - actual_cell_occupancy[each_element])
                            denominator_occ[0, pl] = denominator_occ[0, pl] + abs(actual_cell_occupancy[each_element])

                            numerator_speed[0, pl] = numerator_speed[0, pl] + abs(
                                estimated_cell_speed[each_element] - actual_cell_speed[each_element])
                            denominator_speed[0, pl] = denominator_speed[0, pl] + abs(actual_cell_speed[each_element])

                            numerator_flow[0, pl] = numerator_flow[0, pl] + abs(
                                estimated_cell_flow[each_element] - actual_cell_flow[each_element])
                            denominator_flow[0, pl] = denominator_flow[0, pl] + abs(actual_cell_flow[each_element])

                    for each_cell in range(cell_number):
                        each_cell_error_occ[0, each_cell] = (numerator_occ[0, each_cell] + 0.00000000001) / (
                                    denominator_occ[0, each_cell] + 0.00000000001)
                        each_cell_error_speed[0, each_cell] = (numerator_speed[0, each_cell] + 0.00000000001) / (
                                    denominator_speed[0, each_cell] + 0.00000000001)
                        each_cell_error_flow[0, each_cell] = (numerator_flow[0, each_cell] + 0.00000000001) / (
                                    denominator_flow[0, each_cell] + 0.00000000001)

                total_estimation_error_occ = np.mean(each_cell_error_occ)
                total_estimation_error_speed = np.mean(each_cell_error_speed)
                total_estimation_error_flow = np.mean(each_cell_error_flow)

                combined_error = np.concatenate((each_cell_error_occ, each_cell_error_speed, each_cell_error_flow),
                                                axis=0)
                error_save_to = save_to + "\\cell_error_" + today_date + penetration + "percent.csv"
                np.savetxt(error_save_to, combined_error, delimiter=',')

                print("each_cell_error_occ: ", each_cell_error_occ)
                print(
                "penetration_rate: ", penetration_rate, " total_estimation_error_occ: ", total_estimation_error_occ)

                print("each_cell_error_speed: ", each_cell_error_speed)
                print(
                "penetration_rate: ", penetration_rate, " total_estimation_error_speed: ", total_estimation_error_speed)

                print("each_cell_error_flow: ", each_cell_error_flow)
                print(
                "penetration_rate: ", penetration_rate, " total_estimation_error_flow: ", total_estimation_error_flow)

                fig = plt.figure(figsize=(20, 10), dpi=100)
                yticks = np.arange(1, cell_number + 1)
                sns.set(font_scale=2.5)
                ax = sns.heatmap(KF_occ.transpose(), cmap="RdYlGn_r", annot=False, linewidth=0, rasterized=True, vmin=0,
                                 vmax=35, yticklabels=yticks)
                tit1 = "Cell Occupancy (" + str(int(penetration_rate * 100)) + "% penetration rate)"
                ax.set(title=tit1, xlabel="Time step", ylabel="Cell ID")
                occ_plot_file = save_to + "\\occ_" + today_date + penetration + "percent.png"
                fig.savefig(occ_plot_file)

                fig = plt.figure(figsize=(20, 10), dpi=100)
                yticks = np.arange(1, cell_number + 1)
                ax = sns.heatmap(measurement_speed.transpose(), cmap="RdYlGn", annot=False, linewidth=0,
                                 rasterized=True, vmin=0, vmax=65, yticklabels=yticks)
                tit2 = "Speed (" + str(int(penetration_rate * 100)) + "% penetration rate)"
                ax.set(title=tit2, xlabel="Time step", ylabel="Cell ID")

                speed_plot_file = save_to + "\\speed_" + today_date + penetration + "percent.png"
                fig.savefig(speed_plot_file)

                fig = plt.figure(figsize=(20, 10), dpi=100)
                yticks = np.arange(1, cell_number + 1)
                ax = sns.heatmap(KF_flow.transpose(), cmap="RdYlGn_r", annot=False, linewidth=0, rasterized=True,
                                 vmin=0, vmax=15, yticklabels=yticks)
                tit2 = "Flow (" + str(int(penetration_rate * 100)) + "% penetration rate)"
                ax.set(title=tit2, xlabel="Time step", ylabel="Cell ID")
                flow_plot_file = save_to + "\\flow_" + today_date + penetration + "percent.png"
                fig.savefig(flow_plot_file)

                fig = plt.figure(figsize=(20, 10), dpi=100)
                yticks = np.arange(1, cell_number + 1)
                ax = sns.heatmap(Cell_travel_times.transpose(), cmap="RdYlGn_r", annot=False, linewidth=0,
                                 rasterized=True, vmin=0, vmax=50, yticklabels=yticks)
                tit2 = "Cell Travel Time in Seconds (" + str(int(penetration_rate * 100)) + "% penetration rate)"
                ax.set(title=tit2, xlabel="Time step", ylabel="Cell ID")
                cell_time_plot_file = save_to + "\\cell_time_" + today_date + penetration + "percent.png"
                fig.savefig(cell_time_plot_file)

                for each_time in range(total_time_steps):
                    for each_cell in range(cell_number):
                        Cell_travel_times[each_time, each_cell] = (boundaries[each_cell + 1] - boundaries[
                            each_cell]) / (actual_speed_matrix[each_time, each_cell] * 1.46667)
                    Link_travel_times[each_time, 0] = sum(Cell_travel_times[each_time, :])

                    for j in range(cell_number):
                        if (actual_speed_matrix[each_time, cell_number - j - 1] < 35):
                            Link_queue_length[each_time, 0] = Link_queue_length[each_time, 0] + \
                                                              actual_occ_matrix[each_time][cell_number - j - 1]
                        else:
                            break

                # Link_travel_times = pd.read_csv()
                fig = plt.figure(figsize=(12, 7), dpi=100)
                plt.plot(np.multiply(range(total_time_steps), 6), Link_travel_times[:, 0], linewidth=0.8)
                plt.xlabel('time (sec)')
                plt.ylabel('link travel times (sec)')
                # plt.xlim()
                plt.ylim(0, 100)
                plt.show(fig)
                figure_file1 = save_to + "\\link_travel_time_" + today_date + penetration + ".png"
                fig.savefig(figure_file1)

                travel_time_file = save_to + "\\link_travel_time_" + today_date + penetration + ".csv"
                np.savetxt(travel_time_file, Link_travel_times, delimiter=',')

                fig = plt.figure(figsize=(12, 7), dpi=100)
                plt.plot(np.multiply(range(total_time_steps), 6), Link_queue_length[:, 0], linewidth=0.8)
                plt.xlabel('time (sec)')
                plt.ylabel('link queue length (veh)')
                plt.ylim(0, 40)
                plt.show(fig)
                figure_file2 = save_to + "\\link_queue_length_" + today_date + penetration + ".png"
                fig.savefig(figure_file2)

                queue_length_file = save_to + "\\link_queue_length_" + today_date + penetration + ".csv"
                np.savetxt(queue_length_file, Link_queue_length, delimiter=',')



            except Error as e:
                print('error')

                for pl in range(cell_number):
                    actual_cell_density = list(
                        actual_traffic_state.loc[actual_traffic_state.iloc[:, 1] == pl + 1].iloc[:, 3])
                    if (pl == 2):
                        fig = plt.figure(figsize=(6, 3.5), dpi=100)
                        print("################ Cell", pl + 1, "##################")
                        plt.plot(KF_occ[:, pl], linewidth=0.8)
                        plt.plot(mea_occ[:, pl] * (1.0 / penetration_rate), linewidth=0.6)
                        plt.plot(actual_cell_density, linewidth=0.6)
                        plt.xlabel('time step')
                        plt.ylabel('occupancy (veh)')
                        plt.legend(['Estimated Occupancy', 'Measured Occupancy', 'Actual Density'])
                        plt.show(fig)



    except Error as e:
        conn.close()


def rand(y):
    return np.random.normal(0, y)


def calculate_cell_id(distance_traveled):
    # param: cell_length, distance_traveled#
    cell_id = math.ceil(distance_traveled / cell_length)
    return cell_id


def calculate_bound_locations(link_length, cell_length, cell_number):
    bounds = []
    for i in range(cell_number):
        bounds.append(round(cell_length * i, 1))
    bounds.append(link_length)
    return bounds


def add_data_to_PTS_table_time_space_diagram_method(conn, time, START_TIME, boundaries, epsilon, cell_length):
    '''
    :params: conn: connection; time: current time; START_TIME: simulation start time;
             boundaries: boundary locations of cells; epsilon: buffer size related with location
    :output: no output. Extract probe vehicle traffic states and store them to the database
    '''
    if ((time % time_step) == 0):

        current_time = time
        start_time = START_TIME
        time_step_id = int((time - start_time) / time_step)
        current_probe_traffic_state = np.zeros(shape=(2, cell_number + 1))  ### one cell for storing input flow
        bounds = boundaries
        cur = conn.cursor()
        sql_select_occ = '''SELECT cell_id,COUNT(*) FROM BSM
        WHERE simulation_time = ?
        GROUP BY cell_id'''
        sql_values = (round(current_time, 1),)
        cur.execute(sql_select_occ, sql_values)
        getdata = list(cur.fetchall())
        for each_item in getdata:
            item_id = each_item[0]
            item_occ = each_item[1]
            current_probe_traffic_state[0][item_id] = item_occ

        for i in range(len(bounds)):
            each_bound = bounds[i]
            sql_select_occ = '''SELECT vehicle_id FROM BSM
            WHERE current_pos_section BETWEEN ? AND ?
            AND time_step_id = ?
            GROUP BY vehicle_id'''
            sql_values = (each_bound - epsilon, each_bound + epsilon, time_step_id)
            if (i == 0):
                sql_values = (-5, 5, time_step_id)
            cur.execute(sql_select_occ, sql_values)
            getdata = list(cur.fetchall())
            current_probe_traffic_state[1][i] = len(getdata)

        all_mean_speed = np.zeros(cell_number)
        all_max_speed = np.zeros(cell_number)

        for i in range(cell_number):
            cell_id = i + 1
            ##### get speeds
            sql_select_speed = '''SELECT SUM(current_speed), MAX(current_speed), COUNT(*)
            FROM BSM
            WHERE cell_id = ? 
            AND time_step_id = ?
            '''
            sql_value_speed = (cell_id, time_step_id)
            outflow = current_probe_traffic_state[1][i]
            inflow = current_probe_traffic_state[1][i - 1]
            occ = current_probe_traffic_state[0][i]
            try:
                cur.execute(sql_select_speed, sql_value_speed)
                getspeed = cur.fetchall()
                # print(getspeed)
                if (getspeed[0][2] > 0):
                    mean_speed = getspeed[0][0] / getspeed[0][2] * 1.0
                    max_speed = getspeed[0][1]

                    all_mean_speed[i] = mean_speed
                    all_max_speed[i] = max_speed
                else:
                    mean_speed = -1
                    max_speed = -1

                    all_mean_speed[i] = mean_speed
                    all_max_speed[i] = max_speed

                    # print(mean_speed,max_speed,getspeed[0][2])
                sql_insert = '''INSERT INTO PROBE_TRAFFIC_STATE(time_step_id, cell_id, outflow, inflow, occupancy, mean_speed, max_speed)
                VALUES(?,?,?,?,?,?,?)'''
                insert_values = (time_step_id, cell_id, outflow, inflow, occ, mean_speed, max_speed)
                cur.execute(sql_insert, insert_values)
                # print("insert data to PROBE_TRAFFIC_STATE")
            except Error as e:
                print(e)
        conn.commit()

    if ((((time - START_TIME) % 60.0) == 0) and ((time - START_TIME) != 0)):
        one_min_state = np.zeros(shape=(4, cell_number))
        one_min_id = (time - START_TIME) / 60.0
        cur = conn.cursor()
        sql_get_occ = """
        SELECT vehicle_id FROM BSM
        WHERE cell_id = ? 
        AND simulation_time BETWEEN ? AND ?
        GROUP BY vehicle_id 
        """
        sql_get_flow = """
        SELECT vehicle_id FROM BSM
        WHERE current_pos_section BETWEEN ? AND ?
        AND simulation_time BETWEEN ? AND ?
        GROUP BY vehicle_id
        """
        sql_get_speed = '''
        SELECT SUM(current_speed), MAX(current_speed), COUNT(*)
        FROM BSM
        WHERE cell_id = ? 
        AND simulation_time BETWEEN ? AND ?
        '''
        for cell in range(cell_number):
            sql_values_occ = (cell + 1, time - 60, time)
            cur.execute(sql_get_occ, sql_values_occ)
            getocc = list(cur.fetchall())
            occupancy = len(getocc)
            print("one_min_occ: ", occupancy)
            one_min_state[0][cell] = occupancy

            sql_values_speed = (cell + 1, time - 60, time)
            cur.execute(sql_get_speed, sql_values_speed)
            getspeed = cur.fetchall()
            if (getspeed[0][2] > 0):
                mean_speed = getspeed[0][0] / getspeed[0][2] * 1.0
                max_speed = getspeed[0][1]
                flow = mean_speed * occupancy * 5280 / cell_length
            else:
                mean_speed = -1
                max_speed = -1
                flow = 0
            one_min_state[2][cell] = mean_speed
            one_min_state[3][cell] = max_speed

            sql_insert_one_min = '''INSERT INTO ONE_MIN_STATES(one_min_id, cell_id, occ, flow, mean_speed, max_speed)
                VALUES(?,?,?,?,?,?)'''
            insert_values_one_min = (one_min_id, cell + 1, occupancy * 5280 / cell_length, flow, mean_speed, max_speed)
            cur.execute(sql_insert_one_min, insert_values_one_min)
        conn.commit()

    return [all_mean_speed, all_max_speed, current_probe_traffic_state]


# def Edie_method(conn, time, START_TIME, boundaries, epsilon, cell_length):


def Time_space_data_extraction(conn, time, START_TIME, boundaries, cell_length):
    '''
    :params: conn: connection; time: current time; START_TIME: simulation start time;
             boundaries: boundary locations of cells; epsilon: buffer size related with location
    :output: no output. Vehicle traffic states extracted using Edie's method
    '''
    all_mean_speed = np.zeros(cell_number)
    current_traffic_state = np.zeros(shape=(2, cell_number + 1))
    if ((time % time_step) == 0):
        start_time = START_TIME
        time_step_id = int((time - start_time) / time_step)
        # current_traffic_state = np.zeros(shape=(2, cell_number+1))  ### one cell for storing input flow
        # all_mean_speed = np.zeros(cell_number)
        bounds = boundaries
        cur = conn.cursor()
        data_to_submit = []
        for i in range(cell_number):
            cell_id = i + 1
            sql_get_all_vehicles = """
            SELECT vehicle_id, current_pos_section FROM BSM
            WHERE cell_id = ?
            AND time_step_id = ?              
            """
            sql_values = (cell_id, time_step_id)
            cur.execute(sql_get_all_vehicles, sql_values)
            get_data = pd.DataFrame(list(cur.fetchall()))
            if (len(get_data) > 0):
                get_data.columns = ['vehicle_id', 'distance']
                list_vehicle_id = get_data.vehicle_id.unique()

                list_travel_distances = []
                list_travel_times = []

                for each_vehicle in list_vehicle_id:
                    vehicle_trajectory = get_data[get_data['vehicle_id'] == each_vehicle].iloc[:, 1]
                    get_travel_time = len(vehicle_trajectory) * 0.1 / 3600
                    get_distance = (vehicle_trajectory.iloc[-1] - vehicle_trajectory.iloc[0]) / 5280
                    list_travel_distances.append(get_distance)
                    list_travel_times.append(get_travel_time)

                if (len(list_travel_times) > 0):
                    if (cell_id == cell_number):
                        cell_length_to_use = bounds[-1] - bounds[-2]
                    else:
                        cell_length_to_use = cell_length
                    flow = sum(list_travel_distances) / (
                                (time_step * 1.0 / 3600) * (cell_length_to_use / 5280))  ## the unit is veh per hour
                    density = sum(list_travel_times) / (
                                (time_step * 1.0 / 3600) * (cell_length_to_use / 5280))  ## the unit is veh per mile
                    space_mean_speed = sum(list_travel_distances) / (sum(list_travel_times))

                    current_traffic_state[0][i] = density
                    current_traffic_state[1][i] = flow
                    all_mean_speed[i] = space_mean_speed
                else:
                    flow = 0
                    density = 0
                    space_mean_speed = -1
                    current_traffic_state[0][i] = density
                    current_traffic_state[1][i] = flow
                    all_mean_speed[i] = space_mean_speed
                print("TS: ", time, cell_id, flow, density, space_mean_speed)
                data_to_submit.append((time_step_id, cell_id, flow, density, space_mean_speed))
            else:
                flow = 0
                density = 0
                space_mean_speed = 0
                current_traffic_state[0][i] = density
                current_traffic_state[1][i] = flow
                all_mean_speed[i] = space_mean_speed
                data_to_submit.append((time_step_id, cell_id, flow, density, space_mean_speed))
        sql_insert_ts_state = '''INSERT INTO PROBE_TRAFFIC_STATE_TS(time_step_id, cell_id, flow, density, space_mean_speed) VALUES(?,?,?,?,?)'''
        cur.executemany(sql_insert_ts_state, data_to_submit)
        conn.commit()

        sql_get_input_flow = '''SELECT vehicle_id FROM BSM
        WHERE current_pos_section BETWEEN ? AND ?
        AND time_step_id = ?
        GROUP BY vehicle_id'''
        sql_values_input_flow = (-5, 10, time_step_id)
        sql_values_exit_flow = (bounds[-1] - 10, bounds[-1], time_step_id)

        cur.execute(sql_get_input_flow, sql_values_input_flow)
        get_data = pd.DataFrame(list(cur.fetchall()))
        input_flow_rate = 0
        if (len(get_data) > 0):
            get_data.columns = ['vehicle_id']
            list_vehicle_id = get_data.vehicle_id.unique()
            input_flow_count = len(list_vehicle_id)
        else:
            input_flow_count = 0

        cur.execute(sql_get_input_flow, sql_values_exit_flow)
        get_data = pd.DataFrame(list(cur.fetchall()))
        if (len(get_data) > 0):
            get_data.columns = ['vehicle_id']
            list_vehicle_id = get_data.vehicle_id.unique()
            exit_flow_count = len(list_vehicle_id)
        else:
            exit_flow_count = 0

        print(((((time - START_TIME) % 60.0) == 0) and ((time - START_TIME) != 0)))
        ########## start to calculate one_minute_state #########
    if ((((time - START_TIME) % 60.0) == 0) and ((time - START_TIME) != 0)):
        print("calculate one-minute state")
        one_min_id = int((time - START_TIME) * 1.0 / 60.0)
        time_step_id = int((time - start_time) / time_step)
        cur = conn.cursor()
        sql_get_from_ts_table = """
        SELECT flow, density FROM PROBE_TRAFFIC_STATE_TS
        WHERE time_step_id BETWEEN ? AND ?
        AND cell_id = ?
        """
        current_time_step = time_step_id
        early_time_step = time_step_id - int(60 * 1.0 / time_step) + 1
        data_to_submit_to = []
        for each_cell in range(cell_number):
            each_cell_id = each_cell + 1
            sql_values = (early_time_step, current_time_step, each_cell_id)
            cur.execute(sql_get_from_ts_table, sql_values)
            get_data = list(cur.fetchall())
            list_flows = [i[0] for i in get_data]
            list_densities = [i[1] for i in get_data]
            agg_flow = sum(list_flows) / len(list_flows)
            agg_density = sum(list_densities) / len(list_densities)
            agg_speed = sum(list_flows) / sum(list_densities)
            data_to_submit_to.append((one_min_id, each_cell_id, agg_density, agg_flow, agg_speed))
            print("1_min: ", current_time_step, one_min_id, each_cell_id, agg_density, agg_flow, agg_speed)
        sql_insert_to_new_one_min_table = """INSERT INTO ONE_MIN_STATES_NEW(one_min_id, cell_id, density, flow, mean_space_speed) VALUES(?,?,?,?,?)"""
        try:
            cur.executemany(sql_insert_to_new_one_min_table, data_to_submit_to)
        except Error as e:
            print(e)
        conn.commit()

    if ((time % time_step) == 0):
        print("current time: ", time)
        return [all_mean_speed, current_traffic_state, input_flow_count]


def heatmap_plot(KF_occ):
    plt.figure(figsize=(20, 10), dpi=100)
    yticks = [1, 2, 3, 4, 5, 6, 7]
    ax = sns.heatmap(KF_occ.transpose(), cmap="RdYlGn_r", annot=False, linewidth=0, rasterized=True, vmin=0, vmax=25,
                     yticklabels=yticks)
    ax.set(title="Cell Occupancy", xlabel="Time step", ylabel="Cell ID")


if __name__ == '__main__':
    main()
