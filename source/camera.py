# Define a class to represent the camera
import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
class Camera():
    def __init__(self):
        self.nx = 9
        self.ny = 6
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.
        
        
    def calibrate(self, path = '../data/camera_cal/calibration*.jpg'):
        # Make a list of calibration images
        images = glob.glob(path)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.ny*self.nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx,0:self.ny].T.reshape(-1,2)

        # Step through the list and search for chessboard corners
        i = 0
        for fname in images:
            i += 1
            print('Loading calibration images at ' + str(int(100 * i/(len(images)))) + ' %...')
            img = cv2.imread(fname)
            plt.imshow(img)
            plt.show()
            #plt.show(wait = False)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx,self.ny),None)
        
            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
        
                # Draw and display the corners
                #img = cv2.drawChessboardCorners(img, (self.nx,self.ny), corners, ret)
                #cv2.imshow('img',img)
                #cv2.waitKey(200)

        cv2.waitKey(1)
        cv2.destroyAllWindows()