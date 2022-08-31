import sys
# from turtle import left
import cv2
import numpy as np
import time
import imutils

# Function for stereo vision and depth estimation
import triangulation as tri
import calibration
# Mediapipe for face detection
# import mediapipe as mp
import time



# Stereo vision setup parameters
frame_rate = 30    #Camera frame rate (maximum at 120 fps)
B = 6              #Distance between the cameras [cm]
f = 2.6              #Camera lense's focal length [mm]
alpha = 73     #Camera field of view in the horisontal plane [degrees]

# frame_rate = 120    #Camera frame rate (maximum at 120 fps)
# B = 9               #Distance between the cameras [cm]
# f = 8              #Camera lense's focal length [mm]
# alpha = 56.6     #Camera field of view in the horisontal plane [degrees]

left_pathimg = r"D:\error_calibration\data_tune1\image_L_21.png"
right_pathimg = r"D:\error_calibration\data_tune1\image_R_21.png"

def draw_line(image):
    start_point_col = ( int(image.shape[1]/2),0)
    end_point_col = (int(image.shape[1]/2),int(image.shape[0]))
    color = (0, 255, 0)
    thickness = 1
    start_point_row = (0,int(image.shape[0]/2))
    end_point_row = (int(image.shape[1]),int(image.shape[0]/2))

    cv2.line(image, start_point_col, end_point_col, color, thickness)
    cv2.line(image, start_point_row, end_point_row, color, thickness)

# left_pathimg=r"D:\Doan\Final_project\code\Stereo_vision\StereoVisionDepthEstimation\images\imageL5.png"
# right_pathimg =r"D:\Doan\Final_project\code\Stereo_vision\StereoVisionDepthEstimation\images\imageR5.png"


img_left= cv2.imread(left_pathimg)
img_right= cv2.imread(right_pathimg)
img_right, img_left = calibration.undistortRectify(img_right, img_left)
cv2.imwrite(r"D:\anaconda3\envs\final_project\project\calib_project\undistort.jpg", img_left)
# img_right, img_left = calibration.undistortRectify(img_right, img_left)
draw_line(img_left)
draw_line(img_right)
print("Bouding box image left")
boundBox = cv2.selectROI(img_left)
print(boundBox)
center_point_left = (boundBox[0] +  (boundBox[2] / 2), boundBox[1] +(boundBox[3] / 2))

print("Bouding box image right")
boundBox = cv2.selectROI(img_right)
print(boundBox)
center_point_right = (boundBox[0] +  boundBox[2] / 2, boundBox[1] +boundBox[3] / 2)
print(img_left.shape)
print(center_point_left)
print(center_point_right)



depth = tri.find_depth(center_point_right, center_point_left, img_right, img_left, B, f, alpha)
print(depth)

############
