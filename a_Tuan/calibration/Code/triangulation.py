import sys
import cv2
import numpy as np
import time


def find_depth(right_point, left_point, frame_right, frame_left, baseline,f, alpha):

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = right_point[0]
    x_left = left_point[0]
    y_left = left_point[1]
    # CALCULATE THE DISPARITY:
    disparity = x_left-x_right      #Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    zDepth = (baseline*f_pixel)/disparity             #Depth in [cm]
    X= (zDepth*abs(x_left-width_left/2))/f_pixel
    Y= (zDepth*abs(y_left-height_left/2))/f_pixel
    Xr=(X*4)/3
    Yr=(Y*4)/3
    zDepthr=(zDepth*0.01*4)/3
    return (zDepthr,Xr,Yr)

