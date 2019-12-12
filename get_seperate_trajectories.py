# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:20:21 2019

@author: chen4416
"""

from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy
from scipy import spatial

mypath = "D://radar_trajectory//seperate_vehicle_trajectory"


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


unitsPerPixel = 0.142

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

############### reference point #################
ticks_right_x = np.linspace(min(cp_right[:, 0]), max(cp_right[:, 0]), (max(cp_right[:, 0]) - min(cp_right[:, 0])) * 10)
ticks_right_y = spline_right(ticks_right_x)
ticks_left_x = np.linspace(min(cp_left[:, 0]), max(cp_left[:, 0]), (max(cp_left[:, 0]) - min(cp_left[:, 0])) * 10)
ticks_left_y = spline_left(ticks_left_x)
ticks_middle_x = np.linspace(min(cp_middle[:, 0]), max(cp_middle[:, 0]),
                             (max(cp_middle[:, 0]) - min(cp_middle[:, 0])) * 10)
ticks_middle_y = spline_middle(ticks_middle_x)
ticks_aux_x = np.linspace(min(cp_aux[:, 0]), max(cp_aux[:, 0]), (max(cp_aux[:, 0]) - min(cp_aux[:, 0])) * 10)
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


# listOfFiles = getListOfFiles(mypath)
folder_left = "D:\\radar_trajectory\\seperate_vehicle_trajectory\\Left"
folder_middle = "D:\\radar_trajectory\\seperate_vehicle_trajectory\\Middle"
folder_right = "D:\\radar_trajectory\\seperate_vehicle_trajectory\\Right"
folder_auxiliary = "D:\\radar_trajectory\\seperate_vehicle_trajectory\\Auxilary"
folder_list = [folder_left, folder_middle, folder_right, folder_auxiliary]

for each_folder in folder_list:
    listOfFiles = getListOfFiles(each_folder)
    for each_file in listOfFiles:
        traj1 = pd.read_csv(each_file, sep=",")
        traj = traj1.drop(['Unnamed: 7'], axis=1)
        if (len(traj) > 0):
            vehicle_id = int(each_file.split('\\')[4].split('.')[0])
            if ("Right" in each_file):
                choose_from_x = ticks_right_x
                choose_from_y = ticks_right_y
                choose_from_distance = ticks_right_distance
            elif ("Left" in each_file):
                choose_from_x = ticks_left_x
                choose_from_y = ticks_left_y
                choose_from_distance = ticks_left_distance
            elif ("Middle" in each_file):
                choose_from_x = ticks_middle_x
                choose_from_y = ticks_middle_y
                choose_from_distance = ticks_middle_distance
            else:
                choose_from_x = ticks_aux_x
                choose_from_y = ticks_aux_y
                choose_from_distance = ticks_aux_distance
            for each_row in range(len(traj)):
                x_cord = traj.iloc[each_row][2]
                y_cord = traj.iloc[each_row][3]
                point = [x_cord, y_cord]
                choose_from_cord = zip(choose_from_x, choose_from_y)
                closest_point = choose_from_cord[spatial.KDTree(choose_from_cord).query(point)[1]]
                distance, index = spatial.KDTree(choose_from_cord).query(point)
                traj.at[each_row, "distance"] = choose_from_distance[index]
        # save_dir = each_file.split("\\")[0]+'\\'+each_file.split("\\")[1]+'\\'+"veh_"+str(vehicle_id)+ ".txt"
        save_dir = each_file.split("\\")[0] + '\\' + each_file.split("\\")[1] + '\\' + each_file.split("\\")[2] + '\\' + \
                   each_file.split("\\")[3] + "_vehicle1" + '\\' + "veh_" + str(vehicle_id) + ".txt"
        traj.to_csv(save_dir, sep=',', mode='a')



