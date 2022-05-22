import cv2 as cv


if __name__ == '__main__':
    stereo_cam = cv.VideoCapture(1)
    stereo_cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    stereo_cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    
    n = 0
    
    while(True):
        ret, stereo_frame = stereo_cam.read()

        # cv.imshow('Stereo Camera', stereo_frame)
        
        frameL = stereo_frame[:, :stereo_frame.shape[1]//2, :]
        frameR = stereo_frame[:, stereo_frame.shape[1]//2:, :]

        cv.imshow('Left Camera', frameL)
        cv.imshow('Right Camera', frameR)

        if cv.waitKey(1) == ord('q'):
            break

        if cv.waitKey(1) == ord('s'):
            cv.imwrite('./stereo_images/stereoL/imageL_' + str(n) + '.png', frameL)
            cv.imwrite('./stereo_images/stereoR/imageR_' + str(n) + '.png', frameR)
            
            print(f'Image pair {n} saved successfully!')
            n += 1