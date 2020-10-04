#!/usr/bin/env python3

import os
import sys
import numpy as np
import cv2 as cv
#import scipy.ndimage, imageio

# in opencv, 2D coordinate frames are: x right, y down

# transformation matrices take coordinate vectors (v) and make new coordinate vectors (v')
# M * v = v'

# transformations can be composed by multiplication:
#   M2 * (M1 * v)
# = (M2 * M1) * v

# the order of matrix multiplication matters. M2*M1 != M1*M2

# those vectors are column vectors in the mathematical sense
# in practice, they're simple python tuples (x,y,1) or 1d numpy arrays

# numpy matrices are indexed [row,column] (i.e. [y,x])
# numpy provides the @ infix operator for matrix multiplication
# ordinary * is elementwise

# the matrices we build here assume (x,y,1) column vectors of coordinates

# the (x,y,1) form is called homogeneous coordinates

# affine transformations need homogeneous coordinates to produce translation

# an affine transformation matrix has shape 2x3
# its parts can be understood as the rotation/scaling/shearing part (first two columns)
# and the translation part (third column)

# when transforming between coordinate frames, you can have two senses.
# the "forward" (push) sense maps source coordinates to destination coordinates
# the backward (pull) sense maps destination coordinates to source coordinates

# opencv's warpAffine uses the pull sense. for every destination pixel
# it calculates the point in the source image (and then it interpolates)

# to convert between senses, just invert the matrix.
# if it's a 2x3 matrix, complete it to be a 3x3 martix using row [0,0,1], then invert.

def normalize(vector):
	vector = np.array(vector)
	length = np.linalg.norm(vector)
	return vector / length

def translation(tx, ty):
	result = np.eye(3)
	result[0:2, 2] = (tx, ty) # set last column
	return result

def scale(sx, sy):
	result = np.eye(3)
	result[0,0] = sx # set values on the diagonals
	result[1,1] = sy
	return result

def rotation_by_angle(degrees):
	# let's be lazy
	result = np.eye(3)
	# opencv's rotation sense is +x towards -y, i.e. ccw in a x-right y-down coordinate system
	# I'd like it to be +x towards +y
	# so we could negate the angle
	#result[0:2, 0:3] = cv.getRotationMatrix2D(center=(0,0), angle=-degrees, scale=1.0)

	# let's build this ourselves
	alpha = degrees / 180 * np.pi # in radians
	c = np.cos(alpha)
	s = np.sin(alpha)

	result[0:2, 0] = (c, s) # map x axis to this vector
	result[0:2, 1] = (-s, c)

	return result

def rotation_by_axis(xaxis):
	xaxis = normalize(xaxis)
	(dx, dy) = xaxis

	result = np.eye(3)
	result[0:2, 0] = (dx, dy) # x coordinates are mapped to this
	result[0:2, 1] = (-dy, dx) # y: 90 degree rotated vector

	return result

def shear(xaxis, yaxis):
	result = np.eye(3)
	result[0:2, 0] = xaxis # x coordinates (x,0) are mapped to this
	result[0:2, 1] = yaxis # (0,y) are mapped to this
	return result


### MAIN PROGRAM ###

if len(sys.argv) >= 2:
	srcpath = sys.argv[1]
else:
	srcpath = cv.samples.findFile("lena.jpg")

src = cv.imread(srcpath)
sh, sw = src.shape[:2]

dw, dh = (1024, 768) # of anything you like
#dw, dh = sw, sh

# let's build two transformations

# first, shake and bake
# rotate 30 degrees counterclockwise
# thanks to opencv's interpretation of the value, positive is ccw, which is +x towards -y
# mathematically positive angles rotate +x towards +y
M1 = cv.getRotationMatrix2D(center=(sw/2, sh/2), angle=30, scale=1.0)
# uses same center for source and destination frame
# to center it in an output canvas of a different size you'd have to shift it some more

# second, DIY
# let's use the forward transform sense, i.e. source point transformed into dest point
# p_dest = T2 * R * T1 * p_src
# T1 moves the rotation center in the source to the origin (subtract)
# R rotates (around "origin")
# T2 moves points to the rotation center in the destination (add)

T1 = translation(tx=-sw/2, ty=-sh/2)
T2 = translation(tx=+dw/2, ty=+dh/2)

# try these:
R = rotation_by_angle(degrees=-30) # rotates +x towards +y, i.e. clockwise (mathematical sense, opposite of cv.getRotMat2D)
#R = rotation_by_axis((+8.6, -5)) # normalizes the vector before use. (0.866, -0.5) is approximately -30 degrees
#R = shear(xaxis=(1, -0.1), yaxis=(0.5, 0.5)) # notice how the axes are mapped in the output, y is turned diagonal, x is slightly ascending (negative y)

H = T2 @ R @ T1 # @ is np.dot

# I'm calling it H because a 3x3 matrix of this purpose is generally called homography
# homographies can do perspective transformations

# we built an affine transformation
# so the "homography" part should be (0,0,1), i.e. not used at all
assert np.isclose(a=H[2], b=(0,0,1)).all() # check that the last row is (0,0,1)
M2 = H[0:2, :] # take 2x3 part

# opencv warpAffine and warpPerspective internally operate in a pull sense
# they normally invert the given matrix so that you can simply give it a push sense matrix
# if you pass the WARP_INVERSE_MAP flag, that inversion is _not_ done and the matrix you give is used directly
# example: flags=cv.WARP_INVERSE_MAP | cv.INTER_CUBIC

dest1 = cv.warpAffine(src, M=M1, dsize=(dw, dh), flags=cv.INTER_CUBIC)
dest2 = cv.warpAffine(src, M=M2, dsize=(dw, dh), flags=cv.INTER_CUBIC)

cv.imshow("src", src)
cv.imshow("dest1", dest1)
cv.imshow("dest2", dest2)

cv.waitKey(-1)
cv.destroyAllWindows()
