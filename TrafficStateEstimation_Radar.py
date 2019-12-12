
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
start_time = 0
global end_time
end_time = 300
global time_step
time_step = 6
global link_length
link_length = 633 *3.28084  ###  4649 feet
today_date = "12_1"
penetration = "_100_percent"
actual_state_address = "D://Radar_data_process_10_25//actual_radar_state_10_26.csv"


def main():
    penetration_rate = 1.0
    address = "D://Radar_data_process_10_25//Radar_data_process_10_20//radar_100_percent_12_1_sorted.csv"
    # db_file = "C:\\Users\\Administrator\Dropbox\\BSM Project\\MicroSim_scenario1\\BSM_TSE.sqlite"
    db_file = "D://Radar_data_process_10_25//radar_BSM_12_1_100_percent.sqlite"
    save_to = "D://Radar_data_process_10_25//radar_100_percent_12_1"

    global counter
    counter = 1
    global cell_length
    cell_length = 65 *time_step *5280.0 /3600
    global cell_number
    cell_number = int(math.ceil(link_length /cell_length))  # in feet
    boundaries = calculate_bound_locations(link_length, cell_length, cell_number)


    link_lane_number = 3
    cell_length_in_mile = cell_length /5280.0
    ####### initialize the Kalman Filter #######
    ####### set up matrices to store data #######

    total_time_steps = int((end_time - start_time ) /time_step)
    KF_occ = np.zeros(shape=(total_time_steps, cell_number))      ### the estimated occupancy with KF
    mea_occ = np.zeros(shape=(total_time_steps, cell_number))     ### the measured occupancy
    mea_occ_via_speed = np.zeros(shape=(total_time_steps, cell_number))    ### the occupancy calculated with measured speeds
    pre_occ = np.zeros(shape=(total_time_steps, cell_number))   ### the prediceted occupacny with CTM
    KF_flow = np.zeros(shape=(total_time_steps, cell_number +1))   ### the estimated flow with KF
    mea_flow = np.zeros(shape=(total_time_steps, cell_number +1))    ### measured flow  ???? don't know how to get it ???
    measurement_speed = np.zeros(shape=(total_time_steps, cell_number))   ### measured speed
    KF_speed = np.zeros(shape=(total_time_steps, cell_number))  ### speeds calculated using the estimated occupancy values
    pre_speed = np.zeros(shape=(total_time_steps, cell_number)) ### the predicted speed

    Cell_travel_times = np.zeros(shape=(total_time_steps, cell_number))
    Link_travel_times = np.zeros(shape=(total_time_steps ,1))
    Link_queue_length = np.zeros(shape=(total_time_steps ,1))

    KF_capacity = np.zeros(shape=(total_time_steps, 1))     ### capacity
    KF_capacity[0 ,0] = 2300 *link_lane_number   ### initialized capacity
    KF_ffs = np.zeros(shape=(total_time_steps, 1))   ### free-flow speed
    KF_ffs[0 ,0] = 65    ### initilized speed
    KF_bws = np.zeros(shape=(total_time_steps, 1))   ### bws
    KF_bws[0 ,0] = 16

    # average_car_length = pe.get_average_car_length()
    average_car_length = 15
    jam_density = int(5280 /(average_car_length *1.5))*link_lane_number
    InputFlowRate = 0
    KF_occ_1_min =  np.zeros(shape=(int((end_time -start_time ) *1.0 /60.0), cell_number))
    KF_flow_1_min =  np.zeros(shape=(int((end_time -start_time ) *1.0 /60.0), cell_number))
    KF_speed_1_min =  np.zeros(shape=(int((end_time -start_time ) *1.0 /60.0), cell_number))

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
    P_occ = np.zeros(shape=(total_time_steps ,cell_number))
    P_speed = np.zeros(shape=(total_time_steps ,cell_number))
    P_occ[0 ,:] = [0.9 ] *cell_number
    P_speed[0 ,:] = [0.2 ] *cell_number
    # address = "D://BSM//20%_scenario//Section_6146_20%_45min.csv"


    try:
        with open(address ,'r') as data_csv:
            raw_data = list(csv.reader(data_csv ,delimiter=','))
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
                        if ('Time' in eachLine[1]):
                            continue
                        simulation_time = float(eachLine[1]) - 1506534901.01
                        if (round(simulation_time ,1) == round(t ,1)):
                            counter = counter +1 ## it is not the problem of sqlite
                            vehicle_id = int(eachLine[2])
                            # simulation_time = round(float(eachLine[1]),1)
                            lane_number_str = eachLine[6]
                            if (lane_number_str == 'Left'):
                                lane_number = 1
                            elif (lane_number_str == 'Middle'):
                                lane_number = 2
                            else:
                                lane_number = 3
                            world_pos_x = round(float(eachLine[4]) ,3)
                            world_pos_y = round(float(eachLine[5]) ,3)
                            current_speed =  float(eachLine[9] ) *2.23694
                            current_pos_section = (633 -float(eachLine[3]) )*3.28084 # in the network
                            # speed_drop = pe.get_speed_drop(conn, simulation_time, vehicle_id, current_speed)
                            time_step_id = int(math.ceil((simulation_time -start_time ) /time_step))
                            cell_id = int(calculate_cell_id(abs(current_pos_section)))
                            data = (vehicle_id ,simulation_time ,lane_number ,current_pos_section, \
                                    world_pos_x ,world_pos_y ,current_speed ,time_step_id ,cell_id)

                            data_to_submit.append(data)
                        else:
                            break
                    try:
                        cur.executemany(''' INSERT INTO BSM(vehicle_id,simulation_time,lane_number,
                        current_pos_section, world_pos_x,world_pos_y,current_speed, time_step_id, cell_id)
                        VALUES(?,?,?,?,?,?,?,?,?)''', data_to_submit)

                    except Error as e:
                        print(e)
                    conn.commit()

                    ##### adjust probe traffic state table #####
                    ########start to apply Kalman filter#######
                    if ((round(t ,1) % time_step) == 0):
                        ##### get estimation from the function
                        ##### get input flow #####
                        if (round(t ,1) == start_time):
                            [mean_speeds, probe_traffic_states, input_flow, out_flow] = Time_space_data_extraction(conn, round(t,1), start_time, boundaries, cell_length)
                            occ_measurement = probe_traffic_states[0][1: ]*cell_length /5280
                            print(occ_measurement)
                            occ_measurement[-1] = probe_traffic_states[0][-1 ] * \
                                        (boundaries[-1 ] -boundaries[-2] ) /5280.0
                            print(occ_measurement)
                            flow_measurement = probe_traffic_states[1][: ] *time_step /3600
                            measurement_speed[0 ,:] = mean_speeds
                            mea_occ[0][:] = occ_measurement
                            mea_flow[0][:] = flow_measurement
                            mea_occ_via_speed[0][:] = occ_measurement

                            InputFlowRate = input_flow /penetration_rate
                            OutputFlowRate = out_flow /penetration_rate
                            KF_occ[0 ,:] = mea_occ[0][:]
                            KF_flow[0 ,:] = mea_flow[0][:]
                            P_occ[1 ,:] = P_occ[0 ,:]

                        else:
                            i = int((round(t ,1 ) -start_time ) *1.0 /time_step)
                            print("Predict time step: ", i)
                            [mean_speeds, probe_traffic_states, input_flow, out_flow] = Time_space_data_extraction(conn, round(t,1), start_time, boundaries, cell_length)
                            occ_measurement = probe_traffic_states[0][1: ]*cell_length /5280
                            print(occ_measurement)
                            occ_measurement[-1] = probe_traffic_states[0][-1 ] *(boundaries[-1 ] -boundaries[-2] ) /5280
                            print(occ_measurement)
                            flow_measurement = probe_traffic_states[1][: ]*time_step /3600

                            mea_occ[i][:] = occ_measurement
                            mea_flow[i][:] = flow_measurement

                            InputFlowRate = input_flow /penetration_rate
                            OutputFlowRate = out_flow /penetration_rate
                            print("input flow: ", InputFlowRate)

                            KF_bws[i][0] = 9.35
                            KF_ffs[i][0] = 62.3
                            KF_capacity[i][0] = 1909 *link_lane_number


                            for j in range(cell_number +1):
                                if (j == 0):
                                    KF_flow[i][j] = InputFlowRate

                                elif (j == cell_number):
                                    KF_flow[i][j] = min(KF_occ[ i -1][ j -1], KF_capacity[i][0 ] *time_step /3600, (KF_bws[i][0 ] /KF_ffs[i][0 ] *1.0 ) *
                                                                    (jam_density *(cell_length ) /5280.0 -0))

                                else:
                                    Capacity1 = KF_capacity[i][0]
                                    ffspeed1 = KF_ffs[i][0]
                                    swspeed1 = KF_bws[i][0]
                                    KF_flow[i][j] = min(KF_occ[ i -1][ j -1], Capacity1 *time_step /3600, (swspeed1 /ffspeed1*1.0 ) *(jam_density*(boundaries[ j +1 ] -boundaries[j] ) /5280-KF_occ[i -1][j]))


                            print("mean_speeds: " ,mean_speeds)
                            for j in range(cell_number):  ##the ith cell has j = i-1
                                pre_occ[i][j] = KF_occ[ i -1][j] + KF_flow[i][j] - KF_flow[i][ j +1]
                                if (pre_occ[i][j ] <0):
                                    print ( "Measurement errors: negative occupancy ", j)

                                if (mean_speeds[j] > 0):
                                    if (mean_speeds[j] >= KF_ffs[i][0 ] *0.9):
                                        Occ_mea = mea_occ[i][j ] /penetration_rate

                                    else:
                                        Occ_mea = min(KF_capacity[i][0 ] *(boundaries[ j +1 ] -boundaries[j] ) /
                                                    (5280 *mean_speeds[j]) ,jam_density *KF_bws[i][0 ] *
                                                                  ((boundaries[ j +1 ] -boundaries[j] ) *1.0/5280 ) /
                                                                  (mean_speeds[j ] +KF_bws[i][0]))
                                        print(i ,j, 'measure_occ', Occ_mea)
                                else:

                                    Occ_mea = min(KF_capacity[i][0 ] *(boundaries[ j +1 ] -boundaries[j] ) /
                                                (5280 *KF_speed[ i -1][j]) ,jam_density*KF_bws[i][0 ] *
                                                              ((boundaries[ j +1 ] -boundaries[j] ) *1.0 /5280 ) /
                                                              (KF_speed[ i -1][j ] +KF_bws[i][0]))
                                    # Occ_mea = max( 0, KF_occ[i-1][j] )
                                mea_occ_via_speed[i][j] = Occ_mea

                                P_occ[i][j] =  P_occ[ i -1][j] + Q_occ
                                if (P_occ[i][j] < 0):
                                    print("##########################ERROR################################")
                                ##### the equation for Kalman Gain needs to be modified.
                                ##### should includes H function corresponding to the measurement

                                KalmanGain = P_occ[i][j ] *1.0 /(P_occ[i][j ]+ R_occ)
                                KF_occ[i][j] =  max(0, pre_occ[i][j ]+ KalmanGain *
                                            (mea_occ_via_speed[i][j] - pre_occ[i][j]))
                                P_occ[i][j] = P_occ[i][j] - KalmanGain *P_occ[i][j]

                                ##### calculate KF_speed ####
                                Density = KF_occ[i][j ] /cell_length_in_mile
                                option1 = KF_ffs[i ,0]
                                option2 = KF_capacity[i ,0 ] /Density
                                option3 = (jam_density -Density ) *KF_bws[i, 0 ] /Density
                                pre_speed[i][j] = min(option1, option2, option3) + rand(Q_speed)
                                mea_speed =  mean_speeds[j]
                                P_speed[i][j] = P_speed[i][j] + Q_speed
                                KG_speed =  P_speed[i][j ] /( P_speed[i][j] +R_speed)
                                KF_speed[i][j] = (1 - KG_speed) * pre_speed[i][j] + KG_speed * mea_speed
                                P_speed[i][j] = P_speed[i][j] - KG_speed * P_speed[i][j]

                                if (mean_speeds[j] > 0):
                                    measurement_speed[i, j] = mean_speeds[j]
                                    Cell_travel_times[i, j] = (boundaries[j + 1] - boundaries[j]) / (
                                                mean_speeds[j] * 1.46667)
                                    # Cell_travel_times[i,j] = (boundaries[j+1]-boundaries[j])/(actual_speeds[j]*1.46667)

                                else:
                                    measurement_speed[i, j] = KF_speed[i][j]
                                    Cell_travel_times[i, j] = (boundaries[j + 1] - boundaries[j]) / (
                                                KF_speed[i][j] * 1.46667)
                                    # Cell_travel_times[i,j] = (boundaries[j+1]-boundaries[j])/(KF_speed[i][j]*1.46667)
                                # print("Time Step: ", i, " Cell: ", j+1, " Speed: ", KG_speed[i][j])

                            Link_travel_times[i, 0] = sum(Cell_travel_times[i, :])

                            for j in range(cell_number):
                                if (measurement_speed[i, cell_number - j - 1] < 35):
                                    Link_queue_length[i, 0] = Link_queue_length[i, 0] + KF_occ[i][cell_number - j - 1]
                                else:
                                    break

                        fig = plt.figure(figsize=(10, 3.5), dpi=100)
                        plt.plot(KF_occ[:, 2], linewidth=0.8)
                        plt.plot(mea_occ[:, 2] * (1.0), linewidth=0.3)
                        plt.plot(mea_occ_via_speed[:, 2], linewidth=0.3)
                        plt.plot(pre_occ[:, 2], linewidth=0.3)
                        plt.xlabel('time step')
                        plt.ylabel('occupancy (veh)')
                        plt.legend(
                            ['Estimation', 'Measured Occ via Occ', 'Measured Occ via Speed', 'Predict Occ via CTM'])
                        plt.show(fig)
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

                actual_occ_matrix = np.zeros(shape=(50, cell_number))
                actual_flow_matrix = np.zeros(shape=(50, cell_number))
                actual_speed_matrix = np.zeros(shape=(50, cell_number))

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
                    plt.plot(KF_flow[:, pl + 1], linewidth=0.8)
                    plt.plot(actual_cell_flow, linewidth=0.6)
                    plt.xlabel('time step')
                    plt.ylabel('flow (vph)')
                    plt.legend(['KF_flow', 'actual_cell_flow'])
                    plt.show(fig)

                    estimated_cell_density = KF_occ[:, pl]
                    estimated_cell_speed = measurement_speed[:, pl]
                    estimated_cell_flow = KF_flow[:, pl + 1]

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
                # print(combined_error)
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
                yticks = [1, 2, 3, 4]
                sns.set(font_scale=2.5)
                ax = sns.heatmap(KF_occ.transpose(), cmap="RdYlGn_r", annot=False, linewidth=0, rasterized=True, vmin=0,
                                 vmax=35, yticklabels=yticks)
                tit1 = "Cell Occupancy (" + str(int(penetration_rate * 100)) + "% penetration rate)"
                ax.set(title=tit1, xlabel="Time step", ylabel="Cell ID")

                # plt.xlabel('Years', fontsize = 15)
                occ_plot_file = save_to + "\\occ_" + today_date + penetration + ".png"
                fig.savefig(occ_plot_file)

                fig = plt.figure(figsize=(20, 10), dpi=100)
                yticks = [1, 2, 3, 4]
                ax = sns.heatmap(measurement_speed.transpose(), cmap="RdYlGn", annot=False, linewidth=0,
                                 rasterized=True, vmin=0, vmax=65, yticklabels=yticks)
                tit2 = "Speed (" + str(int(penetration_rate * 100)) + "% penetration rate)"
                ax.set(title=tit2, xlabel="Time step", ylabel="Cell ID")

                speed_plot_file = save_to + "\\speed_" + today_date + penetration + ".png"
                fig.savefig(speed_plot_file)

                fig = plt.figure(figsize=(20, 10), dpi=100)
                yticks = [1, 2, 3, 4]
                ax = sns.heatmap(KF_flow.transpose(), cmap="RdYlGn_r", annot=False, linewidth=0, rasterized=True,
                                 vmin=0, vmax=15, yticklabels=yticks)
                tit2 = "Flow (" + str(int(penetration_rate * 100)) + "% penetration rate)"
                ax.set(title=tit2, xlabel="Time step", ylabel="Cell ID")
                flow_plot_file = save_to + "\\flow_" + today_date + penetration + ".png"
                fig.savefig(flow_plot_file)

                fig = plt.figure(figsize=(20, 10), dpi=100)
                yticks = [1, 2, 3, 4]
                ax = sns.heatmap(Cell_travel_times.transpose(), cmap="RdYlGn_r", annot=False, linewidth=0,
                                 rasterized=True, vmin=0, vmax=50, yticklabels=yticks)
                tit2 = "Cell Travel Time in Seconds (" + str(int(penetration_rate * 100)) + "% penetration rate)"
                ax.set(title=tit2, xlabel="Time step", ylabel="Cell ID")
                cell_time_plot_file = save_to + "\\cell_time_" + today_date + penetration + ".png"
                fig.savefig(cell_time_plot_file)

                for each_time in range(50):
                    for each_cell in range(4):
                        Cell_travel_times[each_time, each_cell] = (boundaries[each_cell + 1] - boundaries[
                            each_cell]) / (actual_speed_matrix[each_time, each_cell] * 1.46667)
                    Link_travel_times[each_time, 0] = sum(Cell_travel_times[each_time, :])

                    for j in range(4):
                        if (actual_speed_matrix[each_time, 4 - j - 1] < 35):
                            Link_queue_length[each_time, 0] = Link_queue_length[each_time, 0] + \
                                                              actual_occ_matrix[each_time][4 - j - 1]
                        else:
                            break

                # Link_travel_times = pd.read_csv()
                fig = plt.figure(figsize=(12, 7), dpi=100)
                plt.plot(np.multiply(range(50), 6), Link_travel_times[:, 0], linewidth=0.8)
                plt.xlabel('time (sec)')
                plt.ylabel('link travel times (sec)')
                # plt.xlim()
                plt.ylim(0, 60)
                plt.show(fig)
                figure_file1 = save_to + "\\link_travel_time_" + today_date + penetration + ".png"
                fig.savefig(figure_file1)

                travel_time_file = save_to + "\\link_travel_time_" + today_date + penetration + ".csv"
                np.savetxt(travel_time_file, Link_travel_times, delimiter=',')

                fig = plt.figure(figsize=(12, 7), dpi=100)
                plt.plot(np.multiply(range(50), 6), Link_queue_length[:, 0], linewidth=0.8)
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
        exit_flow_count = 0
        get_data = pd.DataFrame(list(cur.fetchall()))
        if (len(get_data) > 0):
            get_data.columns = ['vehicle_id']
            list_vehicle_id = get_data.vehicle_id.unique()
            exit_flow_count = len(list_vehicle_id)
        else:
            exit_flow_count = 0

        print(((((time - START_TIME) % 60.0) == 0) and ((time - START_TIME) != 0)))

    if ((time % time_step) == 0):
        print("current time: ", time)
        return [all_mean_speed, current_traffic_state, input_flow_count, exit_flow_count]


if __name__ == '__main__':
    main()
