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




left_pathimg = r"D:\Doan\Final_project\code\Stereo_vision\newcalib\8_6\Left\image_L_2022.png"
right_pathimg = r"D:\Doan\Final_project\code\Stereo_vision\newcalib\8_6\Right\image_R_2022.png"



img_left= cv2.imread(left_pathimg)
img_right= cv2.imread(right_pathimg)
print("************")
print(len(img_left))
img_right, img_left = calibration.undistortRectify(img_right, img_left)
cv2.imwrite(r"D:\Doan\Final_project\code\Stereo_vision\newcalib\undistort_L.jpg", img_left)
cv2.imwrite(r"D:\Doan\Final_project\code\Stereo_vision\newcalib\undistort_R.jpg", img_right)
