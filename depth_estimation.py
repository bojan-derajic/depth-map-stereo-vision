import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import keyboard

if __name__ == '__main__':
    f = 426.6667
    B = 60.0

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

    stereo_cam = cv.VideoCapture(1)

    if not stereo_cam.isOpened():
        print('Cannot open the Camera')
        exit()

    stereo_cam.set(cv.CAP_PROP_FRAME_WIDTH, 2*W)
    stereo_cam.set(cv.CAP_PROP_FRAME_HEIGHT, H)
    
    block_size = 7
    d_min = 10
    d_max = 8*16

    sgbm = cv.StereoSGBM_create(
        minDisparity = d_min,
        numDisparities = d_max,
        blockSize = block_size,
        P1 = 8*2*block_size**2,
        P2 = 32*2*block_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 5,
        speckleWindowSize = 10,
        speckleRange = 1,
        mode = cv.STEREO_SGBM_MODE_SGBM
    )

    fig = plt.figure()
    grid = fig.add_gridspec(2, 2)
    
    ax_stereo = fig.add_subplot(grid[0, :])
    ax_stereo.set_title('Stereo Camera')
    ax_stereo.axis('off')
    ax_disp = fig.add_subplot(grid[1, 0])
    ax_disp.set_title('Disparity Map [pixels]')
    ax_disp.axis('off')
    ax_depth = fig.add_subplot(grid[1, 1])
    ax_depth.set_title('Depth Map [mm]')
    ax_depth.axis('off')

    img_stereo = ax_stereo.imshow(np.zeros((H, 2*W)), vmin=0, vmax=255)
    img_disp = ax_disp.imshow(np.zeros((H, W)), cmap='plasma', vmin=d_min, vmax=d_max)
    img_depth = ax_depth.imshow(np.zeros((H, W)), cmap='plasma_r', vmin=f*B/d_max, vmax=f*B/d_min)

    fig.colorbar(img_disp, ax=ax_disp)
    fig.colorbar(img_depth, ax=ax_depth)

    plt.ion()
    
    while(True):
        ret, stereo_frame = stereo_cam.read()
        
        frameL = cv.cvtColor(stereo_frame[:, :stereo_frame.shape[1]//2, :], cv.COLOR_BGR2RGB)
        frameR = cv.cvtColor(stereo_frame[:, stereo_frame.shape[1]//2:, :], cv.COLOR_BGR2RGB)

        frameL = cv.remap(frameL, stereo_mapL_x, stereo_mapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        frameR = cv.remap(frameR, stereo_mapR_x, stereo_mapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        
        frameL = cv.resize(frameL[y:y+h, x:x+w], (W, H), interpolation=cv.INTER_CUBIC)
        frameR = cv.resize(frameR[y:y+h, x:x+w], (W, H), interpolation=cv.INTER_CUBIC)

        stereo_frame = np.concatenate([frameL, np.zeros((H, 5, 3), dtype='uint8'), frameR], 1)

        frameL_gray = cv.cvtColor(frameL, cv.COLOR_RGB2GRAY)
        frameR_gray = cv.cvtColor(frameR, cv.COLOR_RGB2GRAY)

        disp_map = np.uint8(sgbm.compute(frameL_gray, frameR_gray)/16)
        disp_map = disp_map[:, d_min+d_max:]
        disp_map = cv.medianBlur(disp_map, 5)
        disp_map = cv.bilateralFilter(disp_map, d=0, sigmaColor=2, sigmaSpace=3)

        depth_map = f*B/disp_map

        img_stereo.set_data(stereo_frame)
        img_disp.set_data(disp_map)
        img_depth.set_data(depth_map)
        plt.pause(0.001)

        try:
            if keyboard.is_pressed(' '):
                plt.close()
                plt.ioff()
                stereo_cam.release()

                break
        except:
            pass
    
    plt.show()