#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Calibration with a checkerboard.

Created on Fri Apr  5 10:17:29 2019

@author: tobbe
"""

DIRECTORY = 'calibration-patrick'
SUFFIX = 'jpeg'

# Target image size/scaling. Only set one parameter to non-zero.
SCALE = 0
WIDTH = 640

# Checkerboard Parameters
length = 126 / 5
size = (9,6)

# Taken from / Inspired by https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((size[0]*size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2) # generates all gridpoints: 0,0 1,0 2,0 ... 0,1 1,1 ...
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob(DIRECTORY + '/*.' + SUFFIX)
images.sort()
# Determine the required scaling for the target image size
img = cv.imread(images[0], cv.IMREAD_IGNORE_ORIENTATION | cv.IMREAD_COLOR)
initshape = img.shape
if SCALE:
    scale = SCALE
elif WIDTH:
    scale = WIDTH / initshape[1]
# Find checkerboard points
for fname in images:
    img = cv.imread(fname, cv.IMREAD_IGNORE_ORIENTATION | cv.IMREAD_COLOR)
    if img.shape != initshape:
        raise('Calibration Image sizes differ!')
    # TODO: Find out if the camera image is rotated ... but anyways, we do not
    # know in which direction to rotate to compensate this. Therefore, we
    # need another camera app that enables us to make un-rotated images.
    img = cv.resize(img, (0,0), fx=scale, fy=scale)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, size, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, size, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()

# Get camera calibration parameters
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Get camera matrix for a picture with specific width/height
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Undistort image
undistorted = cv.undistort(img, mtx, dist, None, newcameramtx)

# Crop the image
x, y, w, h = roi
cropped = undistorted[y:y+h, x:x+w]

for i in range(0):
    cv.imshow('img', img)
    cv.waitKey(500)
    cv.imshow('img', undistorted)
    cv.waitKey(500)
    cv.imshow('img', cropped)
    cv.waitKey(500)
cv.destroyAllWindows()

# Determine reprojection error which is a measure for calibration accuracy
mean_error = 0  # Original implementation from the opencv tutorial
mean_error2 = 0 # Tobbe's implementation that determines the reprojection error as mean of all projected pixels
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    print()
    for p0, p1 in zip(imgpoints[i][:,0,:], imgpoints2[:,0,:]):
        print("pixels", p0, " --- projected ", p1)
    # TODO It seems that the following function does not calculate the mean of the
    # reprojection error for this image but instead the euclidean distance between
    # the two image-point-vectors.
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    error2 = 0
    for pixel, reprojected in zip(imgpoints[i][:,0,:], imgpoints2[:,0,:]):
        error2 += cv.norm(pixel, reprojected, cv.NORM_L2) / len(imgpoints2)
    print("error", error)
    print("error2", error2)
    mean_error += error
    mean_error2 += error2
print()
print( "total error (opencv tutorial): {}".format(mean_error/len(objpoints)) )
print( "total error (Tobbe's impl): {}".format(mean_error2/len(objpoints)) )

with open(DIRECTORY + '/calibration-result.txt', 'w') as file:
    file.write('Calibration with scaling factor: ' + str(scale) + '\n')
    file.write('Image size: ' + str(img.shape) + '\n\n')
    file.write('Camera matrix:\n')
    file.write(str(mtx)+'\n\n')
    file.write('Distortion parameters:\n')
    file.write(str(dist))

"""
# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
"""