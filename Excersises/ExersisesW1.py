# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:07:44 2022

@author: Jacob
"""

from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

""" BASIC IMAGE HANDLING"""

# Directory containing data and images
in_dir = "02502Pythondata1/data/"
# X-ray image
im_name = "metacarpals.png"
# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

#printing line just to get space bewteen compile text
print(" ")

#print the shape of the image
print(im_org.shape)

#print the type of the image
print(im_org.dtype)

#show the image
io.imshow(im_org)
plt.title("Metacarpal image")
io.show()

#get lowest and highest pixel values 
print(im_org.min())  # = 32
print(im_org.max())  # = 208

""" COLOR MAPS """

#working with color maps (cmap)
#There are dif cmaps: jet,cool, hot, pink, copper, coolwarm, cubehelix, and terrain.
#more at https://matplotlib.org/stable/tutorials/colors/colormaps.html
io.imshow(im_org, cmap="jet")
plt.title("Metacarpal image (with colormap)")
io.show()

""" GRAY SCALING """

#contrast with imshow
#all pixels below vmin will show black and over vmax will show white (increasing contrast)
io.imshow(im_org, vmin=80, vmax=160)
plt.title("Metacarpal image (with gray level scaling")
io.show()

#automatic optimize contrast so darkest pixel is black and brigthest is white
io.imshow(im_org, vmin=im_org.min(), vmax=im_org.max())
plt.title("Metacarpal image (automatic gray scaling)")
io.show()

""" HISTOGRAM FUNCTIONS """

#show histogram of image (hist() takes 1dimensional array input. ravel()turns 2d array to 1d)
plt.hist(im_org.ravel(), bins=256)
plt.title("Image histogram")
io.show()

#histogram values can be saved 
h = plt.hist(im_org.ravel(), bins=256)

#The value of a given bin can be found by:
bin_no = 100
count = h[0][bin_no]
print(f"There are {count} pixel values in bin {bin_no}")

#Here h is a list of tuples, where in each tuple the first element is the bin count
#and the second is the bin edge. So the bin edges can for example be found by:
bin_left = h[1][bin_no]
bin_right = h[1][bin_no + 1]
print(f"Bin edges: {bin_left} to {bin_right}")
#Here is an alternative way of calling the histogram function:
y, x, _ = plt.hist(im_org.ravel(), bins=256)
io.show()

#finding the most common range of intensities with the histogram:
highestBin = np.argmax(y) #get index of highest bincount
highestBinHeight = y[highestBin] #get its height(/count)
rangeLeft = x[highestBin] #highest bin leftcut
rangeRight = x[highestBin + 1] #highest bin rightcut
print(f"There are {highestBinHeight} pixel values in bin {highestBin}")
print(f"Bin edges: {rangeLeft} to {rangeRight}")

""" PIXEL VALUES AND IMAGE COORDINATE SYSTEMS """
#We are using scikit-image and the image is represented using a NumPy array.
#Therefore, a two-dimensional image is indexed by rows and columns (abbreviated 
#to (row, col) or (r, c)) with (0, 0) at the top-left corner

# find value of a pixel
r = 110
c = 50
im_val = im_org[r, c]
print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

#Since the image is represented as a NumPy array, the usual slicing operations
#can be used. What does this operation do?

#answer: it takes all the first 30 rows of the picture and sets pixel value = 0
im_org[:30] = 0
io.imshow(im_org)
io.show()

#A mask is a binary image of the same size at the original where all values are
# one or zero
# create and show a mask like this:
    # the mask shows white for all px higher than 100 and else 0
mask = im_org > 250
plt.figure()
plt.imshow(mask, cmap = "gray")
plt.show()

""" COLOR IMAGES """

#All color values described by 3 values R, G, B
# read image
imColorName = "ardeche.jpg"
imColor = io.imread(in_dir + imColorName)
io.imshow(imColor)
io.show()
#print dimensions
print(imColor.shape) #= (600, 800)

#print colors {r g b} of pixel in row 100, column 200
print(imColor[100, 200]) #(sky) = 99,166,234 = mostly blue!

#pixel value can be assigned like:
r = 110
c = 90
imColor[r, c] = [255, 0, 0]

#set upper half of image to all green like this
rowHalf = int( imColor.shape[0] / 2)
imColor[:rowHalf] = [0, 255, 0]
io.imshow(imColor)
io.show()

""" MY IMAGE """

#load 
imJbeName = "jbeCam.jpg"
imJbe = io.imread(in_dir + imJbeName)
io.imshow(imJbe)
io.show()



















