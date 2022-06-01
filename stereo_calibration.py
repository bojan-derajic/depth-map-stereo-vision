from distutils.log import error
import cv2 as cv
import numpy as np
import os
from tqdm import tqdm


if __name__ == '__main__':
    pathL = './stereo_images/stereoL/'
    pathR = './stereo_images/stereoR/'
    
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0.001)
    
    M = 10
    N = 7
    square_size = 22     # [mm]

    objp = np.zeros((M*N, 3), np.float32)
    objp[:, :2] = np.mgrid[0:M, 0:N].T.reshape(-1, 2) * square_size
    
    img_ptsL = []
    img_ptsR = []
    obj_pts = []
    
    n_imgs = len([name for name in os.listdir(pathL)])
    
    if n_imgs == 0:
        error('Cannot find any calibration images.')
        exit()
    
    for i in tqdm(range(n_imgs)):
        imgL = cv.imread(pathL + f'imageL_{i}.png')
        imgR = cv.imread(pathR + f'imageR_{i}.png')
        
        imgL_gray = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        imgR_gray = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    
        retL, cornersL = cv.findChessboardCorners(imgL, (M, N), None)
        retR, cornersR = cv.findChessboardCorners(imgR, (M, N), None)
    
        if retL and retR:
            cv.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
            cv.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
            
            cv.drawChessboardCorners(imgL, (M, N), cornersL, retL)
            cv.drawChessboardCorners(imgR, (M, N), cornersR, retR)
    
            cv.imshow('Stereo Calibration', np.concatenate([imgL, imgR], 1))
            cv.waitKey(100)
    
            obj_pts.append(objp)
            img_ptsL.append(cornersL)
            img_ptsR.append(cornersR)
    
    cv.destroyAllWindows() 
    
    hL, wL = imgL_gray.shape[:2]
    retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(obj_pts, img_ptsL, (wL, hL), None, None)
    opt_mtxL, roiL = cv.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 0, (wL, hL))
    
    hR, wR = imgR_gray.shape[:2]
    retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(obj_pts, img_ptsR, (wR, hR), None, None) 
    opt_mtxR, roiR = cv.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 0, (wR, hR))
    
    flags = cv.CALIB_FIX_INTRINSIC
    
    criteria_stereo = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0.001)
    
    retS, opt_mtxL, distL, opt_mtxR, distR, Rot, Trns, E_mat, F_mat = cv.stereoCalibrate(
        obj_pts, img_ptsL, img_ptsR, opt_mtxL, distL, opt_mtxR, distR, (wL, hL), criteria_stereo, flags 
    )
    
    rectify_scale = 1
    rectL, rectR, proj_matL, proj_matR, Q, roiL, roiR = cv.stereoRectify(
        opt_mtxL, distL, opt_mtxR, distR, (wL, hL), Rot, Trns, rectify_scale, (0, 0)
    )
    
    stereo_mapL = cv.initUndistortRectifyMap(opt_mtxL, distL, rectL, proj_matL, (wL, hL), cv.CV_16SC2)
    stereo_mapR = cv.initUndistortRectifyMap(opt_mtxR, distR, rectR, proj_matR, (wR, hR), cv.CV_16SC2)
    
    print('Saving parameters...')
    cv_file = cv.FileStorage('stereo_cam_map.xml', cv.FILE_STORAGE_WRITE)
    cv_file.write('stereo_mapL_x', stereo_mapL[0])
    cv_file.write('stereo_mapL_y', stereo_mapL[1])
    cv_file.write('stereo_mapR_x', stereo_mapR[0])
    cv_file.write('stereo_mapR_y', stereo_mapR[1])
    cv_file.write('roiL', roiL)
    cv_file.write('roiR', roiR)
    cv_file.write('img_size', (wL, hL))
    cv_file.release()
    
    imgL = cv.imread(pathL + 'imageL_0.png')
    imgR = cv.imread(pathR + 'imageR_0.png')
    stereo_img_raw = np.concatenate([imgL, imgR], 1)
    
    imgL_rect = cv.remap(imgL, stereo_mapL[0], stereo_mapL[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    imgR_rect = cv.remap(imgR, stereo_mapR[0], stereo_mapR[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    
    x = max(roiL[0], roiR[0])
    y = max(roiL[1], roiR[1])
    w = min(roiL[2], roiR[2])
    h = min(roiL[3], roiR[3])

    imgL_rect = cv.resize(imgL_rect[y:y+h, x:x+w], (wL, hL), interpolation=cv.INTER_CUBIC)
    imgR_rect = cv.resize(imgR_rect[y:y+h, x:x+w], (wL, hL), interpolation=cv.INTER_CUBIC)

    stereo_img_rect = np.concatenate([imgL_rect, imgR_rect], 1)
    
    cv.imshow('Raw stereo image', stereo_img_raw)
    cv.imshow('Rectified stereo image', stereo_img_rect)
    cv.waitKey(0)