# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:07:44 2022

@author: Jacob
"""

from skimage import color, io, measure, img_as_ubyte, exposure
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
mask = im_org > 100
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

#examine size
print(imJbe.shape)
#downscale
scaledJbe = rescale(imJbe, 0.25, anti_aliasing=True, channel_axis=2)
#examine dimensions again
print(scaledJbe.shape)

#Try to find a way to automatically scale your image so the resulting width
#(number of columns) is always equal to 400, no matter the size of the input
#image?

scaleFactor = 400/imJbe.shape[1]
scaledJbe = rescale(imJbe, scaleFactor, anti_aliasing=True, channel_axis=2)
print(scaledJbe.shape)

#transform into gray-level image
imGray = color.rgb2gray(scaledJbe)
imByte = img_as_ubyte(imGray)
io.imshow(imByte)
io.show()
#We are forcing the pixel type back into unsigned bytes using the img as ubyte
#function, since the rgb2gray functions returns the pixel type as floating point
#numbers.

#Compute and show histogram of own image
plt.hist(imByte.ravel(), bins=256)
plt.title("JBEGRAY histogram")
io.show()

#load dark and bright histograms
bright = io.imread(in_dir + "bright.jpg")
scaleFactor = 400/bright.shape[1]
bright = rescale(bright, scaleFactor, anti_aliasing=True, channel_axis=2)
bright = color.rgb2gray(bright)
bright = img_as_ubyte(bright)
io.imshow(bright)
io.show()
plt.hist(bright.ravel(), bins=256)
plt.title("Bright histogram")
io.show()

dark = io.imread(in_dir + "dark.jpg")
scaleFactor = 400/dark.shape[1]
dark = rescale(dark, scaleFactor, anti_aliasing=True, channel_axis=2)
dark = color.rgb2gray(dark)
dark = img_as_ubyte(dark)
io.imshow(dark)
io.show()
plt.hist(dark.ravel(), bins=256)
plt.title("Dark histogram")
io.show()

#differences are that bright has a lot of lines to the right (more high values
#dark has all the lines to the left instead... (low values)

""" split RGB channels"""
#load dtu sign. 
dtu = io.imread(in_dir + "DTUSign1.jpg")
io.imshow(dtu)
io.show()
#get the red values
dtu_red = dtu[:, :, 0]
io.imshow(dtu_red)
io.show()
#get green values
dtu_green = dtu[:, :, 1]
io.imshow(dtu_green)
io.show()
#get blue values
dtu_blue = dtu[:, :, 2]
io.imshow(dtu_blue)
io.show()


""" Simple image manipulation """
#create black rectangle in image using image slicing
# set pixels that are in (row 500 to 1000 AND column 800 to 1500) to 0
dtu[500:1000, 800:1500, :] = 0
io.imshow(dtu)
io.show()

#save to disk
#io.imsave("DTUSign1_marked.jpg", dtu)

#make blue rectangle on the sign
dtu[1500:1750, 2200:2800, :] = [0, 0, 255]
io.imshow(dtu)
io.show()

#turn gray image into RGB with color.gray2rgb. Use mask to set bones blue!
im_org = io.imread(in_dir + "metacarpals.png")
io.imshow(im_org)
io.show()
im_org = exposure.rescale_intensity(im_org, (108, 150)) #autocontrast
io.imshow(im_org)
io.show()
im_rgb = color.gray2rgb(im_org)
mask = im_org > 108
im_rgb[mask] = [0, 0, 255]
io.imshow(im_rgb)
io.show()

im_org = io.imread(in_dir + "metacarpals.png")
p = profile_line(im_org, (342, 77), (320, 160))
plt.plot(p)
plt.ylabel("Intensity")
plt.xlabel("Distance along line")
plt.show()



im_org = io.imread(in_dir + "road.png")
im_gray = color.rgb2gray(im_org)
ll = 200
im_crop = im_gray[40:40 + ll, 150:150 + ll]
xx, yy = np.mgrid[0:im_crop.shape[0], 0:im_crop.shape[1]]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xx, yy, im_crop, rstride=1, cstride=1, 
                       cmap=plt.cm.jet,linewidth=0)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()







