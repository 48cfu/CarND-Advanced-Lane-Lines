import numpy as np
import cv2
import glob
import matplotlib
matplotlib.use('Qt4Agg')
#import pyplotlab as plt
#valid strings are ['Qt4Agg', 'WXAgg', 'CocoaAgg', 'cairo', 'agg', 'MacOSX', 
#'GTKAgg', 'template', 'pgf', 'pdf', 'nbAgg', 'GTK3Cairo', 'GTKCairo', 'ps', 
#'Qt5Agg', 'TkAgg', 'WX', 'gdk', 'GTK', 'GTK3Agg', 'svg', 'WebAgg', 'emf']
#matplotlib.use('agg')
import matplotlib.pyplot as plt
#%matplotlib qt
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('../data/camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
i = 0
for fname in images:
    i += 1
    print(i)
    img = cv2.imread(fname)
    plt.imshow(img)
    plt.show(block=False)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        cv2.imshow('img',img)
        cv2.waitKey(200)


cv2.waitKey(0)
cv2.destroyAllWindows()

