# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:40:45 2022

@author: Jacob Berg Eriksen
"""

import math

#Exercise 1
#Explain how to calculate the angle O when a and b is given in the figure below.
# Calculate  (in degrees) when a = 10 and b = 3 using the function math.atan2(). 
# Remember to import math and find out what atan2 does

a = 10 #length
b = 3 #height
angle_rad = math.atan2(b, a)
angle_deg = math.degrees(angle_rad) # 16.7 degrees

#Exercise 2
#Create a Python function called camera_b_distance.
def camera_b_distance(f, g):
    """
    camera_b_distance returns the distance (b) where the CCD should be placed
    when the object distance (g) and the focal length (f) are given
    :param f: Focal length
    :param g: Object distance
    :return: b, the distance where the CCD should be placed
    DISTANCES ARE IN MILIMETERS!
    based on gauss lens equation: 1/g + 1/b = 1/f
    """
    if f*g != 0 and f != g:
        return -(f*g)/(f-g)
    else:
        return 0

#Use your function to find out where the CCD should be placed when the focal 
# length is 15 mm and the object distance is 0.1, 1, 5, and 15 meters.
print(camera_b_distance(15, 100)) #17.647058823529413
print(camera_b_distance(15, 1000)) #15.228426395939087
print(camera_b_distance(15, 5000)) #15.045135406218655
print(camera_b_distance(15, 15000)) #15.015015015015015
#what happens when the object distance is increased?
#- the ccd distance is reduced but less and less the longer the object dist is!

#Exercise 3
