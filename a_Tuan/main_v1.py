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

left_pathimg = r"D:\Doan\Final_project\code\Stereo_vision\newcalib\8_6\Left\image_L_2022.png"
right_pathimg = r"D:\Doan\Final_project\code\Stereo_vision\newcalib\8_6\Right\image_R_2022.png"

# left_pathimg=r"D:\Doan\Final_project\code\Stereo_vision\StereoVisionDepthEstimation\images\imageL5.png"
# right_pathimg =r"D:\Doan\Final_project\code\Stereo_vision\StereoVisionDepthEstimation\images\imageR5.png"


img_left= cv2.imread(left_pathimg)
img_right= cv2.imread(right_pathimg)
print("************")
print(len(img_left))
img_right, img_left = calibration.undistortRectify(img_right, img_left)
cv2.imwrite(r"D:\Doan\Final_project\code\Stereo_vision\newcalib\undistort_L.jpg", img_left)
cv2.imwrite(r"D:\Doan\Final_project\code\Stereo_vision\newcalib\undistort_R.jpg", img_right)
print("###################################################################################################")
print("###################################################################################################")
print(img_left)

# img_right, img_left = calibration.undistortRectify(img_right, img_left)
print("Bouding box image left")
boundBox = cv2.selectROI(img_left)
print(boundBox)
center_point_left = (boundBox[0] +  (boundBox[2] / 2), boundBox[1] +(boundBox[3] / 2))

print("Bouding box image right")
boundBox = cv2.selectROI(img_right)
print(boundBox)
center_point_right = (boundBox[0] +  boundBox[2] / 2, boundBox[1] +boundBox[3] / 2)
print(img_left.shape)
# print(center_point_left)
# print(center_point_right)



depth = tri.find_depth(center_point_right, center_point_left, img_right, img_left, B, f, alpha)
print(depth)

############
