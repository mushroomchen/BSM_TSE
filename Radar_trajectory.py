# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 20:31:34 2019

@author: chen4416
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
from scipy import spatial


def pixels2cart(pixelsXY, unitsPerPixel):
    """Translate the pixel coordinates in pixelsXY to cartesian coordinates from image of scale unitsPerPixel.
       This inverts the Y coordinate to switch from image convention (positive Y down) to traditional (positive Y up)."""
    # pixelsXY = dtype(pixelsXY)
    return np.array([(x * unitsPerPixel, -y * unitsPerPixel) for x, y in pixelsXY])


def computeSpline(controlPoints, degree):
    if len(controlPoints) > degree:
        xSorted = controlPoints[:, 0]
        ySorted = controlPoints[:, 1]
        # splineObj = scipy.interpolate.UnivariateSpline(xSorted, ySorted, k=degree, s=1)
        splineObj = scipy.interpolate.interp1d(xSorted, ySorted)
        return splineObj


def computeSpline1(controlPoints, degree):
    if len(controlPoints) > degree:
        xSorted = controlPoints[:, 0]
        ySorted = controlPoints[:, 1]
        splineObj = scipy.interpolate.UnivariateSpline(xSorted, ySorted, k=degree, s=1)
        # splineObj = scipy.interpolate.interp1d(xSorted, ySorted)
        return splineObj


file1 = "D:\Radar_trajectory\_20170927_1200-1215_all.csv"
file2 = "D:\Radar_trajectory\_20170927_1215-1230_all2.csv"
file3 = "D:\Radar_trajectory\_20190927_1230-1245_all2.csv"
file4 = "D:\Radar_trajectory\_20170927_1245-1300_all2.csv"
column_names = ["row_id", "teimstamp", "count", "object_id", "sensor_id", "x", "y", "vx", "vy", "v", "length",
                "global_x", "global_y", "latitude", "longitude", "lane", "unique_id"]
dt1 = pd.read_csv(file1)
dt1.columns = column_names
dt1['distance'] = np.nan
dt2 = pd.read_csv(file2)
dt2.columns = column_names
dt2['distance'] = np.nan
dt3 = pd.read_csv(file3)
dt3.columns = column_names
dt3['distance'] = np.nan
dt4 = pd.read_csv(file4)
dt4.columns = column_names
dt4['distance'] = np.nan
dt = [dt1, dt2, dt3, dt4]
### for I-94_SiteAerial_Tight.jpg ####
### pixel setting ###
latitude = 44.966610
longitude = -93.271308333
unitsPerPixel = 0.142

controlPoints_aux_fog = [(4689, 378), (4411, 379), (3547, 379), (3107, 379), (2982, 379), (2900, 378), (2722, 362),
                         (2660, 356), (2533, 345), (2357, 328), (2196, 320), (2103, 320), (1845, 320), (1719, 321),
                         (1682, 320), (1595, 309),
                         (1515, 285), (1438, 248), (1359, 192), (1301, 137), (1253, 84), (1217, 23)]

controlPoints_aux_lane = [(4685, 401), (4627, 404), (4275, 404), (3790, 405), (3302, 405), (2887, 404), (2797, 402),
                          (2700, 394),
                          (2535, 379), (2360, 367), (2145, 357), (1925, 357), (1725, 357), (1680, 354), (1575, 344),
                          (1460, 304), (1392, 269), (1317, 214), (1240, 132), (1177, 37)]

controlPoints_right_lane = [(4685, 430), (4457, 429), (4027, 432), (3705, 432), (3290, 432), (2955, 432), (2527, 431),
                            (2321, 431),
                            (1884, 431), (1663, 431), (1554, 431), (1453, 432), (1344, 429), (1239, 422), (1128, 410),
                            (935, 386), (714, 343), (607, 320), (504, 292),
                            (395, 260), (303, 232), (196, 195), (97, 163), (3, 133)]

controlPoints_middle_lane = [(4681, 455), (4031, 455), (3277, 456), (2850, 457), (2520, 458), (2304, 458), (1875, 458),
                             (1771, 457),
                             (1663, 458), (1556, 458), (1447, 457), (1237, 448), (1128, 439), (1020, 424), (935, 410),
                             (708, 369), (600, 345), (496, 320),
                             (393, 287), (290, 255), (183, 219), (82, 187), (3, 161)]

controlPoints_left_lane = [(4689, 481), (3892, 483), (3313, 486), (2373, 486), (2124, 488), (1665, 488), (1560, 488),
                           (1451, 486),
                           (1336, 483), (1231, 475), (1141, 466), (1018, 451), (920, 436), (710, 396), (596, 372),
                           (465, 336), (345, 301), (226, 261),
                           (106, 224), (5, 188)]

cp_right = pixels2cart(controlPoints_right_lane, unitsPerPixel)
cp_left = pixels2cart(controlPoints_left_lane, unitsPerPixel)
cp_middle = pixels2cart(controlPoints_middle_lane, unitsPerPixel)
cp_aux = pixels2cart(controlPoints_aux_lane, unitsPerPixel)

spline_right = computeSpline(cp_right, 3)
spline_left = computeSpline(cp_left, 3)
spline_middle = computeSpline(cp_middle, 3)
spline_aux = computeSpline(cp_aux, 3)

x_right = cp_right[:, 0]
y_right = spline_right(x_right)
x_left = cp_left[:, 0]
y_left = spline_left(x_left)
x_middle = cp_middle[:, 0]
y_middle = spline_middle(x_middle)
x_aux = cp_aux[:, 0]
y_aux = spline_aux(x_aux)

###### check the accuracy of the spline ######
plt.figure(figsize=(10, 5))
plt.plot(cp_right[:, 0], cp_right[:, 1], 'x', cp_right[:, 0], y_right, 'o')
plt.plot(cp_left[:, 0], cp_left[:, 1], 'x', cp_left[:, 0], y_left, 'o')
plt.plot(cp_middle[:, 0], cp_middle[:, 1], 'x', cp_middle[:, 0], y_middle, 'o')
plt.plot(cp_aux[:, 0], cp_aux[:, 1], 'x', cp_aux[:, 0], y_aux, 'o')
plt.figure()

############### reference point #################
ticks_right_x = np.linspace(min(cp_right[:, 0]), max(cp_right[:, 0]), (max(cp_right[:, 0]) - min(cp_right[:, 0])) * 5)
ticks_right_y = spline_right(ticks_right_x)
ticks_left_x = np.linspace(min(cp_left[:, 0]), max(cp_left[:, 0]), (max(cp_left[:, 0]) - min(cp_left[:, 0])) * 5)
ticks_left_y = spline_left(ticks_left_x)
ticks_middle_x = np.linspace(min(cp_middle[:, 0]), max(cp_middle[:, 0]),
                             (max(cp_middle[:, 0]) - min(cp_middle[:, 0])) * 5)
ticks_middle_y = spline_middle(ticks_middle_x)
ticks_aux_x = np.linspace(min(cp_aux[:, 0]), max(cp_aux[:, 0]), (max(cp_aux[:, 0]) - min(cp_aux[:, 0])) * 5)
ticks_aux_y = spline_aux(ticks_aux_x)

ticks_x = [ticks_left_x, ticks_middle_x, ticks_right_x, ticks_aux_x]
ticks_y = [ticks_left_y, ticks_middle_y, ticks_right_y, ticks_aux_y]

# =============================================================================
# plt.figure(figsize=(10,5))
# plt.plot(ticks_right_x, ticks_right_y, 'o')
# plt.plot(ticks_left_x, ticks_left_y, 'o')
# plt.plot(ticks_middle_x, ticks_middle_y, 'o')
# plt.plot(ticks_aux_x, ticks_aux_y, 'o')
# plt.figure()
# =============================================================================
#################################################
ticks_right_distance = [0]
ticks_left_distance = [0]
ticks_middle_distance = [0]
ticks_aux_distance = [0]
ticks_distance = [ticks_left_distance, ticks_middle_distance, ticks_right_distance, ticks_aux_distance]

for each_lane in range(len(ticks_distance)):
    for i in range(1, len(ticks_x[each_lane])):
        distance_between_points = math.sqrt((ticks_x[each_lane][i] - ticks_x[each_lane][i - 1]) ** 2 + (
                    ticks_y[each_lane][i] - ticks_y[each_lane][i - 1]) ** 2)
        ticks_distance[each_lane].append(ticks_distance[each_lane][i - 1] + distance_between_points)

for each_table in range(3, 4):
    current_table = dt[each_table]
    counter = 0
    for each_row in range(len(current_table)):
        x_cord = current_table.iloc[each_row]["global_x"]
        y_cord = current_table.iloc[each_row]["global_y"]
        which_lane = current_table.iloc[each_row]["lane"]
        choose_from_x = []
        choose_from_y = []
        choose_from_distance = []
        if (which_lane == "right_lane"):
            choose_from_x = ticks_x[2]
            choose_from_y = ticks_y[2]
            choose_from_distance = ticks_distance[2]
        elif (which_lane == "left_lane"):
            choose_from_x = ticks_x[0]
            choose_from_y = ticks_y[0]
            choose_from_distance = ticks_distance[0]
        elif (which_lane == "middle_lane"):
            choose_from_x = ticks_x[1]
            choose_from_y = ticks_y[1]
            choose_from_distance = ticks_distance[1]
        else:
            choose_from_x = ticks_x[3]
            choose_from_y = ticks_y[3]
            choose_from_distance = ticks_distance[3]
        point = [x_cord, y_cord]
        choose_from_cord = zip(choose_from_x, choose_from_y)
        closest_point = choose_from_cord[spatial.KDTree(choose_from_cord).query(point)[1]]
        distance, index = spatial.KDTree(choose_from_cord).query(point)
        dt[each_table].at[each_row, "distance"] = choose_from_distance[index]
        counter = counter + 1
        if (counter % 1000 == 0):
            print("finish", counter)

# dt1.to_csv('D:\Radar_trajectory\_20170927_1200-1215_all_with_distance2.csv')
# dt2.to_csv('D:\Radar_trajectory\_20170927_1215-1230_all_with_distance2.csv')
# dt3.to_csv('D:\Radar_trajectory\_20170927_1230-1245_all_with_distance2.csv')
dt4.to_csv('D:\Radar_trajectory\_20170927_1245-1300_all_with_distance2.csv')

# =============================================================================
# file22 = "D:\Radar_trajectory\_20170927_1215-1230_all_with_distance2.csv"
# dt22 = pd.read_csv(file22)
# counter  = 0
# for each_row in range(len(dt22)):
#     if (dt22.iloc[each_row]["distance"] == 0):
#         x_cord = dt22.iloc[each_row]["global_x"]
#         y_cord = dt22.iloc[each_row]["global_y"]
#         which_lane = dt22.iloc[each_row]["lane"]
#         choose_from_x = []
#         choose_from_y = []
#         choose_from_distance = []
#         if (which_lane == "right_lane"):
#             choose_from_x = ticks_x[2]
#             choose_from_y = ticks_y[2]
#             choose_from_distance = ticks_distance[2]
#         elif (which_lane == "left_lane"):
#             choose_from_x = ticks_x[0]
#             choose_from_y = ticks_y[0]
#             choose_from_distance = ticks_distance[0]
#         elif (which_lane == "middle_lane"):
#             choose_from_x = ticks_x[1]
#             choose_from_y = ticks_y[1]
#             choose_from_distance = ticks_distance[1]
#         else:
#             choose_from_x = ticks_x[3]
#             choose_from_y = ticks_y[3]
#             choose_from_distance = ticks_distance[3]
#         point = [x_cord, y_cord]
#         choose_from_cord = zip(choose_from_x,choose_from_y)
#         closest_point = choose_from_cord[spatial.KDTree(choose_from_cord).query(point)[1]]
#         distance, index = spatial.KDTree(choose_from_cord).query(point)
#         dt22.at[each_row,"distance"] = choose_from_distance[index]
#         counter= counter+1
#         print("finish", counter)
#
# dt22.to_csv('D:\Radar_trajectory\_20170927_1215-1230_all_with_distance3.csv')
# =============================================================================
