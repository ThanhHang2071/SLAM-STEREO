import sys
import glob
import cv2 
import numpy as np
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
from matplotlib import pyplot as plt 

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=1 sync=0"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def calib(dir_path, square_size=1, board_size=(7,6)):
    '''
    Calibrate single camera to obtain camera intrinsic parameters from saved frames
    '''

    # Criteria used by checkerboard pattern detector.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Chessboard size
    rows = board_size[0]
    cols = board_size[1]
    world_scaling = square_size #change this to the real world square size. Or not.

    # Coordinates of squares in the checkerboard world space
    
    print('Start cpmpute objp')
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:cols].T.reshape(-1,2)
    objp = world_scaling* objp



    imgpointsL = [] # 2d points in image plane. Pixel coordinates of checkerboards
    imgpointsR = []
    objpoints = [] # 3d point in real world space. Coordinates of the checkerboard in checkerboard world space.

    imagesLeft = glob.glob(dir_path + '/left/*.jpg')
    imagesRight = glob.glob(dir_path + '/right/*.jpg')

    for imgLeft, imgRight in zip(imagesLeft, imagesRight):

        imgL = cv2.imread(imgLeft)
        imgR = cv2.imread(imgRight)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray L', grayL)
        # cv2.imshow('gray R', grayR)
        # cv2.waitKey(0)

        heightL, widthL, _ = imgL.shape
        heightR, widthR, _ = imgR.shape
        

        if heightL != heightR:
            print("Left height ", heightL)
            print("Right height ", heightR)
            print("Two imgs are not the same size")
            break

        # Find the chess board corners  
        print('Start find the chess board corners ', imgLeft)
        retL, cornersL = cv2.findChessboardCorners(grayL, (rows,cols),  cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
        retR, cornersR = cv2.findChessboardCorners(grayR, (rows,cols),  cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
        imgL_good = []
        imgR_good = []


        if retL and retR:
            print("---------------------Good image: ", imgLeft,"--------------------")
            imgL_good.append(imgLeft)
            imgR_good.append(imgRight)

            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            cornersL = cv2.cornerSubPix(grayL, cornersL, conv_size, (-1,-1), criteria)
            imgpointsL.append(cornersL)

            cornersR = cv2.cornerSubPix(grayR, cornersR, conv_size, (-1,-1), criteria)
            imgpointsR.append(cornersR)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(imgL, (rows,cols), cornersL, retL)
            cv2.imshow('img left', imgL)
            cv2.drawChessboardCorners(imgR, (rows,cols), cornersR, retR)
            cv2.imshow('img left', imgL)
            cv2.waitKey(30)

            objpoints.append(objp)
            return imgL_good, imgR_good
      
        

    cv2.destroyAllWindows()

    if len(objpoints) == len(imgpointsL) and len(objpoints) == len(imgpointsR) and len(objpoints) > 0:
        retL, cmtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, (widthL, heightL), None, None)
        print("Left lens: \n")
        print('rmse:', retL)
        print('camera matrix:\n', cmtxL)
        print('distortion coeffs:', distL)

        retR, cmtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, (widthR, heightR), None, None)
        print("Right lens: \n")
        print('rmse:', retR)
        print('camera matrix:\n', cmtxR)
        print('distortion coeffs:', distR)
    else:
        print(len(objpoints))
        print(len(imgpointsL))
        print(len(imgpointsR))
        sys.exit("not enough points to calibrate")

    return cmtxL, distL, cmtxR, distR, imgL_good, imgR_good

def save_frame():
    '''
    Take frames pair from stereo camera
    '''

    cap0 = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0, flip_method=2), cv2.CAP_GSTREAMER)
    # cap1 = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1, flip_method=2), cv2.CAP_GSTREAMER)

    # if cap0.isOpened() and cap1.isOpened():
    if cap0.isOpened():
        i =0
        while True:

            ret0, frame0 = cap0.read() # left
            # ret1, frame1 = cap1.read() # right

            if not ret0:
                break

            cv2.imshow('test0',frame0)
            # cv2.imshow('test1', frame1)

            k = cv2.waitKey(1)

            if k == ord('q'):
                break

            # If space-bar is pressed, save the image
            if k == ord('s'):
                cv2.imwrite(f'images/left/{i}.jpg', frame0)
                # cv2.imwrite(f'images/right/{i}.jpg', frame1)
            i+=1

    else:
        print("Can't open cameras")

    cap0.release()
    # cap1.release()
    cv2.destroyAllWindows()


def showCamRemap():
 
    cv_file = readFileMap()

    stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = getNodes(cv_file)
    

    cap0 = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0, flip_method=2), cv2.CAP_GSTREAMER)
    cap1 = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1, flip_method=2), cv2.CAP_GSTREAMER)
    # cap0 = cv2.VideoCapture(0)


    # while(cap1.isOpened() and cap0.isOpened()):
    while(cap0.isOpened()):

        frame0_remap, frame0, frame1_remap, frame1 = frame_remap(cap0, cap1, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y)
        # Show the frames
        cv2.imshow("frame left remap", frame0_remap)
        cv2.imshow("frame right remap", frame1_remap) 
        cv2.imshow("frame left", frame0)
        cv2.imshow("frame right", frame1)


        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    

    
    # Release and destroy all windows before termination
    # cap1.release()
    cap0.release()

    cv2.destroyAllWindows()


# Downsamples image x number (reduce_factor) of times. 
def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image


def readFileMap():
    # Camera parameters to undistort and rectify images
    cv_file = cv2.FileStorage()
    cv_file.open('stereoMap.xml', cv2.FileStorage_READ)
    print('Read file successfully')
    return cv_file


def getNodes(cv_file):
    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
    
    return stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y

def frame_remap(cap0, cap1, imgRight, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y):
    ret0, frame0 = cap0.read() # left
    ret1, frame1 = cap1.read() # right

    # Undistort and rectify images
    frame0_remap = cv2.remap(frame0, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame1_remap = cv2.remap(frame1, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    return frame0_remap,  frame0, frame1_remap,  frame1

def image_remap(imgLeft, imgRight, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y):
    imgL = cv2.imread(imgLeft) # left
    imgR = cv2.imread(imgRight) # right

    # Undistort and rectify images
    imgL_remap = cv2.remap(imgL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    imgR_remap = cv2.remap(imgR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    return imgL_remap,  imgL, imgR_remap,  imgR


def DisparityMap(imgLeft, imgRight):
    
    
    cv_file = readFileMap()

    stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = getNodes(cv_file)
    
    
    imgL_remap,  imgL, imgR_remap,  imgR = image_remap(imgLeft, imgRight, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y)
    
    # # Show the frames
    # cv2.imshow("frame right", imgR) 
    # cv2.imshow("frame left", imgL)

    # cv2.waitKey(0)
    
    
    # Show the frames
    # cv2.imshow("frame left undistorted", imgL_remap)
    # cv2.imshow("frame right undistorted", imgR_remap) 
    # cv2.waitKey(0)
    
    # Downsample each image 3 times (because they're too big)
    imgL = downsample_image(imgL_remap,3)
    imgR = downsample_image(imgR_remap,3)

    imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # # Show the frames
    # cv2.imshow("frame right downscaled", imgR) 
    # cv2.imshow("frame left downscaled", imgL)
    # cv2.waitKey(0)
    
    #=========================================================
    # Create Disparity map from Stereo Vision
    #=========================================================

    # For each pixel algorithm will find the best disparity from 0
    # Larger block size implies smoother, though less accurate disparity map

    # Set disparity parameters
    # Note: disparity range is tuned according to specific parameters obtained through trial and error. 
    block_size = 5
    min_disp = -1
    max_disp = 31
    num_disp = max_disp - min_disp # Needs to be divisible by 16

    # Create Block matching object. 
    stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
        numDisparities = num_disp,
        blockSize = block_size,
        uniquenessRatio = 5,
        speckleWindowSize = 5,
        speckleRange = 2,
        disp12MaxDiff = 2,
        P1 = 8 * 3 * block_size**2,#8*img_channels*block_size**2,
        P2 = 32 * 3 * block_size**2) #32*img_channels*block_size**2)


    #stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=win_size)

    # Compute disparity map
    disparity_map = stereo.compute(imgLgray, imgRgray)

    # # Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
    # plt.imshow(disparity_map,'gray')
    # plt.show()

    # #=========================================================
    # # Generate Point Cloud from Disparity Map
    # #=========================================================

    # Get new downsampled width and height 
    h,w = imgR.shape[:2]
    f = 0.8 * w  # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                    [0, 0, 0, -f],  # so that y-axis looks up
                    [0, 0, 1, 0]])

    # Convert disparity map to float32 and divide by 16 as show in the documentation
    print("disparity_map.dtype : ", disparity_map.dtype)
    disparity_map = np.float32(np.divide(disparity_map, 16.0))
    print("disparity_map.dtype : ", disparity_map.dtype)

    # Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)
    # Get color of the reprojected points
    colors = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

    # Get rid of points with value 0 (no depth)
    mask_map = disparity_map > disparity_map.min()

    # Mask colors and points. 
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]
    
    return output_points, output_colors


def create_point_cloud_file(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')



if __name__ == '__main__':
    # save_frame()

    imgs_folder = 'images'
    # intrinsic_matrix_Left, distort_Left, intrinsic_matrix_Right, distort_Right = calib(imgs_folder, square_size=2.8, board_size=(4,3))
    # main(intrinsic_matrix_Left, distort_Left)
    # main(intrinsic_matrix_Left, distort_Left)

    # readFileMap()

    # showCamRemap()
    
    ## Image choose
    imgLeft = imgs_folder + "/left/601.jpg"
    imgRight = imgs_folder + "/right/601.jpg"
    
    
    output_points, output_colors = DisparityMap(imgLeft, imgRight)
    print("Done")
    output_file = 'pointCloud.ply'
    # Generate point cloud file
    create_point_cloud_file(output_points, output_colors, output_file)





    