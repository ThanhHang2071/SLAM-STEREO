# Reconstructing 3D point cloud from two stereo images

<details open>
<summary style="font-weight:500;font-size:20px">Approach</summary>

1. Collect or take stereo images.
- Preprocessing (camera calibration and image rectification): 
    - Intrinsic camera: focal lengths, central coordinate, valid pixels, distortion coefficients.
    - Extrinsic camera: rotation matrix, translation vectors.
- Algorithm: planar rectification
2. Feature extraction
3. Feature matching 
4. Disparity calculation (Disparity Map)
- The disparity is the apparent motion of objects between a pair of stereo images. 
![disparity map](https://www.baeldung.com/wp-content/uploads/sites/4/2022/05/fig1.png)

*Disparity is the horizontal displacement of a point's projections between the left and the right image. Whereas, depth refers to the z coordinate (usually z) of a point located in the real 3D world (x, y, z). Then the disparity map can be converted into a depth map using triangulation*

5. Depth calculation (Depth Map)
- A depth map is a picture where every pixel has depth information (rather than RGB) and it normally represented as a grayscale picture. Depth information means the distance of surface of scene objects from a viewpoint.
- Algorithm: Triangulation

</details>

<details>
<summary style="font-weight:500;font-size:20px">References</summary>

[Disparity Map in Stereo Vision](https://www.baeldung.com/cs/disparity-map-stereo-vision)
[Camera Calibration using OpenCV](https://learnopencv.com/camera-calibration-using-opencv/)
[How to Calibrate your ZED camera with OpenCV](https://www.stereolabs.com/docs/opencv/calibration/)

</details>

https://github.com/robz/OpenCV_Matplotlib_Tests/blob/master/test_checkerboard.py

https://python.hotexamples.com/site/file?hash=0xb1c733adb6198150f5d30ba16e5d219ca2ea6018ff457139adeda4e36c9df43e

https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615