import cv2 as cv


if __name__ == '__main__':
    stereo_cam = cv.VideoCapture(1)
    
    if not stereo_cam.isOpened():
        print('Cannot open the Camera')
        exit()

    stereo_cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    stereo_cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    
    n = 0
    
    while(True):
        ret, stereo_frame = stereo_cam.read()
        
        frameL = stereo_frame[:, :stereo_frame.shape[1]//2, :]
        frameR = stereo_frame[:, stereo_frame.shape[1]//2:, :]

        cv.imshow('Left Camera', frameL)
        cv.imshow('Right Camera', frameR)

        if cv.waitKey(1) == ord('q'):
            break

        if cv.waitKey(1) == ord(' '):
            cv.imwrite(f'./stereo_images/stereoL/imageL_{n}.png', frameL)
            cv.imwrite(f'./stereo_images/stereoR/imageR_{n}.png', frameR)
            
            print(f'Image pair {n} saved successfully!')
            n += 1

    stereo_cam.release()
    cv.destroyAllWindows()