import cv2 as cv

# stereoW, stereoH = (2560, 720)
stereoW, stereoH = (1280, 480)


if __name__ == '__main__':
    stereo_cam = cv.VideoCapture(1)
    
    if not stereo_cam.isOpened():
        print('Cannot open the Camera')
        exit()

    stereo_cam.set(cv.CAP_PROP_FRAME_WIDTH, stereoW)
    stereo_cam.set(cv.CAP_PROP_FRAME_HEIGHT, stereoH)
    
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