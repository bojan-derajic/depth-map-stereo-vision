import cv2 as cv
from cv2 import RETR_LIST
import numpy as np
import os
from tqdm import tqdm


if __name__ == '__main__':
    pathL = './stereo_images/stereoL'
    pathR = './stereo_images/stereoR'

    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0.001)

    M = 7
    N = 6
    objp = np.zeros((M*N, 3), np.float32)
    objp[:, :2] = np.mgrid[0:M, 0:N].T.reshape(-1, 2)
    
    img_ptsL = []
    img_ptsR = []
    obj_pts = []

    n_imgs = len([name for name in os.listdir(pathL)])

    for i in tqdm(range(n_imgs)):
        obj_pts.append(objp)

        imgL = cv.imread(pathL + f'/imageL_{i}.png')
        imgR = cv.imread(pathR + f'/imageR_{i}.png')
        
        imgL_gray = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        imgR_gray = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        retL, cornersL = cv.findChessboardCorners(imgL, (M, N), None)
        retR, cornersR = cv.findChessboardCorners(imgR, (M, N), None)

        if retL and retR:
            obj_pts.append(objp)

            cv.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
            cv.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
            
            cv.drawChessboardCorners(imgL, (M, N), cornersL, retL)
            cv.drawChessboardCorners(imgR, (M, N), cornersR, retR)

            cv.imshow('Stereo Calibration', np.concatenate([imgL, imgR], 1))
            cv.waitKey(0)

            img_ptsL.append(cornersL)
            img_ptsR.append(cornersR)

    hL, wL = imgL_gray.shape[:2]
    retL, mtxL, distL, rvecsL, tvecs = cv.calibrateCamera(obj_pts, img_ptsL, (wL, hL), None, None)
    opt_mtxL, roiL = cv.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

    hR, wR = imgR_gray.shape[:2]
    retR, mtxR, distR, rvecsR, tvecs = cv.calibrateCamera(obj_pts, img_ptsR, (wR, hR), None, None)
    opt_mtxR, roiR = cv.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC

    criteria_stereo = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0.001)

    retS, opt_mtxL, distL, opt_mtxR, distR, Rot, Trns, Emat, Fmat = cv.stereoCalibrate(
        obj_pts, img_ptsL, img_ptsR, opt_mtxL, distL, opt_mtxR, distR, (wL, hL), criteria_stereo, flags 
    )
    
    rectify_scale = 1
    rectL, rectR, proj_matL, proj_matR, Q, roiL, roiR = cv.stereoRectify(
        opt_mtxL, distL, opt_mtxR, distR, (wL, hL), Rot, Trns, rectify_scale, (0, 0)
    )

    stereo_mapL = cv.initUndistortRectifyMap(opt_mtxL, distL, rectL, proj_matL, (wL, hL), cv.CV_16SC2)
    stereo_mapR = cv.initUndistortRectifyMap(opt_mtxR, distR, rectR, proj_matR, (wR, hR), cv.CV_16SC2)

    print('Saving parameters...')
    cv_file = cv.FileStorage('improved_params2.xml', cv.FILE_STORAGE_WRITE)
    cv_file.write('Left_Stereo_Map_x', stereo_mapL[0])
    cv_file.write('Left_Stereo_Map_y', stereo_mapL[1])
    cv_file.write('Right_Stereo_Map_x', stereo_mapR[0])
    cv_file.write('Right_Stereo_Map_y', stereo_mapR[1])
    cv_file.release()