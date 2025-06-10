import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob

# Function to read individual camera calibration
def load_calibration(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            K = np.load(f)  # intrinsics
            R = np.load(f)  # extrinsics orientation Rodrigues
            T = np.load(f)  # extrinsics position
            RT = np.load(f)  # extrinsics matrix 3x4
            dist = np.load(f)  # distortion
            mean_error = np.load(f)  # retroprojection mean error
            H = np.load(f)
            h_error = np.load(f)
    else:
        # Default calibration parameters
        K = np.eye(3)  # intrinsics
        R = np.asarray([[0.], [0.], [0.]])  # extrinsics orientation Rodrigues
        T = np.asarray([[0.], [0.], [0.]])  # extrinsics position
        RT = np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])  # extrinsics matrix 3x4
        dist = np.zeros((1, 5))  # np.asarray([[0., 0., 0., 0., 0.]])  # distortion
        mean_error = np.array(0.)  # retroprojection mean error
        H = np.eye(3)
        h_error = np.array(0.)
    return K, RT, dist

def load_stereo_calibration(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            R = np.load(f)
            T = np.load(f)
            E = np.load(f)
            F = np.load(f)
            ret = True
    else:
        ret = False
        R,E,F = np.zeros(3)
        T = np.zeros((3,1))

    return ret, R, T, E, F

def get_corners(image, pattern_size):
    """Find chessboard corners in image."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, pattern_size)
    if ret == True:
        cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                        (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS,
                        30, 0.01))
    return ret, corners

def add_pair_pts(pattern_size):
    objps, lettps, rightps = [], [], []

    images_left = glob("./capturas_left/*.jpg")
    idx_left = len(images_left)

    images_right = glob("./capturas_right/*.jpg")
    idx_right = len(images_right)

    if idx_left == idx_right:

        objp = np.zeros((1, 6 * 8, 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

        for name1, name2 in zip(images_left, images_right):
            img1 = cv.imread(name1)
            ret1, leftp = get_corners(img1, pattern_size)

            img2 = cv.imread(name2)
            ret2, rightp = get_corners(img2, pattern_size)

            if ret1 and ret2:
                objps.append(objp)
                lettps.append(leftp)
                rightps.append(rightp)

    if idx_left == 0:
        img_shape = (0,0)
        print("Error: There isn't calibration images...")
    else:
        img_shape = img1.shape[:2][::-1]

    return objps, lettps, rightps, img_shape

def calibrate_stereo_cameras():

    # Load individual left camera calibration
    cal_l = np.load('calibration_left.npz')
    cal_r = np.load('calibration_right.npz')
    K_l = cal_l['mtx']
    dist_l = cal_l['dist']

    # Load individual right camera calibration
    K_r= cal_r['mtx']
    dist_r = cal_r['dist']

    pattern_size = (8, 6)
    objps, lettps, rightps, img_shape = add_pair_pts(pattern_size)

    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)
    flags = (cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_FIX_INTRINSIC)
    # flags = cv.CALIB_FIX_INTRINSIC

    ret, K1, D1, K2, D2, R, T, E, F = cv.stereoCalibrate(objps,
                                                         lettps,
                                                         rightps,
                                                         K_l,
                                                         dist_l,
                                                         K_r,
                                                         dist_r,
                                                         img_shape,
                                                         criteria=criteria,
                                                         flags=flags)

    print("Stereo calibration rms: ", ret)
    np.savez("calibration_stereo_data.npz", K1=K1, D1=D1, K2=K2, D2=D2,  R=R, T=T, E=E, F=F)

    return ret, R, T, E, F

if __name__ == "__main__":

    # Load camera calibration
    # ret, R, T, E, F = load_stereo_calibration("./imgs/camera_STEREO/chessboard_9x6/stereo_calib.npy")
    # print('R:',R)
    # print('T:',T)
    # print('E:',E)
    # print('F:',F)

    # Stereo camera calibration
    ret, R, T, E, F = calibrate_stereo_cameras()
    print('R:',R)
    print('T:',T)
    print('E:',E)
    print('F:',F)