import cv2 as cv
import numpy as np


if __name__ == '__main__':
    cv_file = cv.FileStorage('stereo_cam_map.xml', cv.FILE_STORAGE_READ)

    if cv_file.isOpened():
        stereo_mapL_x = cv_file.getNode('stereo_mapL_x').mat()
        stereo_mapL_y = cv_file.getNode('stereo_mapL_y').mat()
        stereo_mapR_x = cv_file.getNode('stereo_mapR_x').mat()
        stereo_mapR_y = cv_file.getNode('stereo_mapR_y').mat()
        roiL = np.squeeze(np.int32(cv_file.getNode('roiL').mat()))
        roiR = np.squeeze(np.int32(cv_file.getNode('roiR').mat()))
        W, H = np.squeeze(np.int32(cv_file.getNode('img_size').mat()))
        cv_file.release()

    x = max(roiL[0], roiR[0])
    y = max(roiL[1], roiR[1])
    w = min(roiL[2], roiR[2])
    h = min(roiL[3], roiR[3])

    stereo_cam = cv.VideoCapture(0)

    if not stereo_cam.isOpened():
        print('Cannot open the Camera')
        exit()

    stereo_cam.set(cv.CAP_PROP_FRAME_WIDTH, 2*W)
    stereo_cam.set(cv.CAP_PROP_FRAME_HEIGHT, H)

    while(True):
        ret, stereo_frame = stereo_cam.read()
        
        frameL = stereo_frame[:, :stereo_frame.shape[1]//2, :]
        frameR = stereo_frame[:, stereo_frame.shape[1]//2:, :]

        frameL = cv.remap(frameL, stereo_mapL_x, stereo_mapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        frameR = cv.remap(frameR, stereo_mapR_x, stereo_mapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        
        frameL = cv.resize(frameL[y:y+h, x:x+w], (W, H), interpolation=cv.INTER_CUBIC)
        frameR = cv.resize(frameR[y:y+h, x:x+w], (W, H), interpolation=cv.INTER_CUBIC)

        stereo_frame = np.concatenate([frameL, frameR], 1)

        cv.imshow('Stereo Camera', stereo_frame)
        
        if cv.waitKey(1) == ord('q'):
            break

    stereo_cam.release()
    cv.destroyAllWindows()

        