import sys
import glob
import cv2 
import numpy as np

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

    imagesLeft = glob.glob(dir_path + '/test_left/*.jpg')
    imagesRight = glob.glob(dir_path + '/test_right/*.jpg')

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
        heightL, widthL, channelsL = imgL.shape
        newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cmtxL, distL, (widthL, heightL), 1, (widthL, heightL))

        print("Left lens: \n")
        print('rmse:', retL)
        print('camera matrix:\n', cmtxL)
        print('distortion coeffs:', distL)

        retR, cmtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, (widthR, heightR), None, None)
        heightR, widthR, channelsR = imgR.shape
        newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cmtxR, distR, (widthR, heightR), 1, (widthR, heightR))

        print("Right lens: \n")
        print('rmse:', retR)
        print('camera matrix:\n', cmtxR)
        print('distortion coeffs:', distR)

        ########## Stereo Vision Calibration #############################################

        # flags = 0
        # flags |= cv.CALIB_FIX_INTRINSIC
        # # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
        # # Hence intrinsic parameters are the same 

        # criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
        # retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)




        # ########## Stereo Rectification #################################################

        # rectifyScale= 1
        # rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

        # stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
        # stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

        # print("Saving parameters!")
        # cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

        # cv_file.write('stereoMapL_x',stereoMapL[0])
        # cv_file.write('stereoMapL_y',stereoMapL[1])
        # cv_file.write('stereoMapR_x',stereoMapR[0])
        # cv_file.write('stereoMapR_y',stereoMapR[1])

        # cv_file.release()

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
    cap1 = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1, flip_method=2), cv2.CAP_GSTREAMER)

    if cap0.isOpened() and cap1.isOpened():
        i =0
        while True:

            ret0, frame0 = cap0.read() # left
            ret1, frame1 = cap1.read() # right

            if not ret0 or not ret1:
                break

            cv2.imshow('test0',frame0)
            cv2.imshow('test1', frame1)

            k = cv2.waitKey(1)

            if k == ord('q'):
                break

            # If space-bar is pressed, save the image
            if k == ord('s'):
                cv2.imwrite(f'images/test_left/{i}.jpg', frame0)
                cv2.imwrite(f'images/test_right/{i}.jpg', frame1)
            i+=1

    else:
        print("Can't open cameras")

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

def main(mtx, dist) :
    img = cv2.imread('images/left/*.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    cv2.imshow('testimg', dst)
    cv2.waitKey(0)
    cv2.imwrite('calibresult.png', dst)

    cv2.destroyAllWindows()


def showCamRemap():
    # Camera parameters to undistort and rectify images
    cv_file = cv2.FileStorage()
    cv_file.open('stereoMap.xml', cv2.FileStorage_READ)
    print('Opened')

    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

    # Open both cameras
    # cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)                    
    # cap_left =  cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap0 = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0, flip_method=2), cv2.CAP_GSTREAMER)
    cap1 = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1, flip_method=2), cv2.CAP_GSTREAMER)



    while(cap1.isOpened() and cap0.isOpened()):

        ret0, frame0 = cap0.read() # left
        ret1, frame1 = cap1.read() # right

        # Undistort and rectify images
        frame1_remap = cv2.remap(frame1, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        frame0_remap = cv2.remap(frame0, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
                        
        # Show the frames
        cv2.imshow("frame right remap", frame1_remap) 
        cv2.imshow("frame left remap", frame0_remap)
        # cv2.imshow("frame left", frame0)
        # cv2.imshow("frame right", frame1)

        # k = cv2.waitKey(1)


        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    
    # Release and destroy all windows before termination
    cap1.release()
    cap0.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # save_frame()

    # imgs_folder = 'images'
    # intrinsic_matrix_Left, distort_Left, intrinsic_matrix_Right, distort_Right = calib(imgs_folder, square_size=2.8, board_size=(4,3))
    # main(intrinsic_matrix_Left, distort_Left)
    # main(intrinsic_matrix_Left, distort_Left)

    showCamRemap()
    