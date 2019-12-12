import sys

sys.path.append('<directory path of Import_BSM_AIMSUN.py>')
import numpy as np
import math
import pandas as pd
import sqlite3
import csv
from sqlite3 import Error

global start_time
start_time = 0
global end_time
end_time = 300
global time_step
time_step = 6
global link_length
link_length = 633 * 3.28084  ###  4649 feet
today_date = "10_27"
penetration = "_100_percent"


# actual_state_address = "D://BSM//create_partial_percentage_10_1//actual_traffic_state_10_2.csv"

def main():
    global counter
    counter = 1
    global cell_length
    # cell_length = pe.initialize_ffs()*time_step*5280.0/3600     #in feet
    cell_length = 70 * time_step * 5280.0 / 3600
    global cell_number
    cell_number = int(math.ceil(link_length / cell_length))  # in feet
    # all_data = pd.read_csv(address)  use pandas package
    # list(all_data.columns.values)
    boundaries = calculate_bound_locations(link_length, cell_length, cell_number)
    address = "C://Users//chen4416.AD//Dropbox//Radar_dara_process_10_20//radar_100_percent_10_20_sorted.csv"
    db_file = "D://Radar_dara_process_10_25//radar_BSM_10_27_100_percent.sqlite"

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
                        simulation_time = float(eachLine[1]) - 1506534901.01
                        if (round(simulation_time, 1) == round(t, 1)):
                            counter = counter + 1  ## it is not the problem of sqlite
                            vehicle_id = int(eachLine[2])
                            # simulation_time = round(float(eachLine[1]),1)
                            lane_number_str = eachLine[6]
                            if (lane_number_str == 'Left'):
                                lane_number = 1
                            elif (lane_number_str == 'Middle'):
                                lane_number = 2
                            else:
                                lane_number = 3
                            world_pos_x = round(float(eachLine[4]), 3)
                            world_pos_y = round(float(eachLine[5]), 3)
                            current_speed = float(eachLine[9]) * 2.23694
                            current_pos_section = (633 - float(eachLine[3])) * 3.28084  # in the network
                            # speed_drop = pe.get_speed_drop(conn, simulation_time, vehicle_id, current_speed)
                            time_step_id = int(math.ceil((simulation_time - start_time) / time_step))
                            cell_id = int(calculate_cell_id(abs(current_pos_section)))
                            data = (vehicle_id, simulation_time, lane_number, current_pos_section, \
                                    world_pos_x, world_pos_y, current_speed, time_step_id, cell_id)

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
                    if ((round(t, 1) % time_step) == 0):
                        ##### get estimation from the function
                        ##### get input flow #####
                        if (round(t, 1) == start_time):
                            [mean_speeds, probe_traffic_states, input_flow] = Time_space_data_extraction(conn,
                                                                                                         round(t, 1),
                                                                                                         start_time,
                                                                                                         boundaries,
                                                                                                         cell_length)


                        else:
                            i = int((round(t, 1) - start_time) * 1.0 / time_step)
                            print("Predict time step: ", i)
                            [mean_speeds, probe_traffic_states, input_flow] = Time_space_data_extraction(conn,
                                                                                                         round(t, 1),
                                                                                                         start_time,
                                                                                                         boundaries,
                                                                                                         cell_length)

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
        get_data = pd.DataFrame(list(cur.fetchall()))
        if (len(get_data) > 0):
            get_data.columns = ['vehicle_id']
            list_vehicle_id = get_data.vehicle_id.unique()
            exit_flow_count = len(list_vehicle_id)
        else:
            exit_flow_count = 0

        print(((((time - START_TIME) % 60.0) == 0) and ((time - START_TIME) != 0)))
        ########## start to calculate one_minute_state #########

    if ((time % time_step) == 0):
        print("current time: ", time)
        return [all_mean_speed, current_traffic_state, input_flow_count]


if __name__ == '__main__':
    main()
