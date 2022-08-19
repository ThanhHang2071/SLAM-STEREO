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
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:cols].T.reshape(-1,2)
    objp = world_scaling* objp

    imgpoints = [] # 2d points in image plane. Pixel coordinates of checkerboards
    objpoints = [] # 3d point in real world space. Coordinates of the checkerboard in checkerboard world space.

    for img in dir_path: 

        img = cv2.imread(img, 1)
        gray_img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width, _ = img.shape

        # Find the chess board corners
        # ret, corners = cv2.findChessboardCorners(gray_img, (rows,cols), None)
        ret, corners = cv2.findChessboardCorners(gray_img, (rows,cols),  cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:

            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            corners2 = cv2.cornerSubPix(gray_img,corners, conv_size, (-1,-1), criteria)
            
            # Draw and display the corners
            # cv2.drawChessboardCorners(img, (rows,cols), corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(5)

            objpoints.append(objp)
            imgpoints.append(corners)

    cv2.destroyAllWindows()

    if len(objpoints) == len(imgpoints) and len(objpoints) > 0:
        ret, cmtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
        print('rmse:', ret)
        print('camera matrix:\n', cmtx)
        print('distortion coeffs:', dist)
    else:
        sys.exit("not enough points to calibrate")

    return cmtx, dist

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
            cv2.waitKey(0)

            # k = cv2.waitKey(1)

            # if  k == ord('q'):
            #     break

            # # If space-bar is pressed, save the image
            # if k == 32:
            #     cv2.imwrite(f'images/left/{i}.jpg', frame0)
            #     cv2.imwrite(f'images/right/{i}.jpg', frame1)
            # i+=1

    else:
        print("Can't open cameras")

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

def main(mtx, dist) :
    img = cv2.imread('images/left/12.jpg')
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


if __name__ == '__main__':
    save_frame()

    # imgs_folder = glob.glob('images/left/*.jpg')
    # intrinsic_matrix, distort = calib(imgs_folder, square_size=2.8, board_size=(4,3))
    # main(intrinsic_matrix, distort)





    