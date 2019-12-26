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
        
        
    def calibrate(self, path = '../data/camera_cal/calibration*.jpg', plot_figures = False):
        # Make a list of calibration images
        images = glob.glob(path)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.ny*self.nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx,0:self.ny].T.reshape(-1,2)

        # Step through the list and search for chessboard corners
        i = 0
        if plot_figures == True:
            fig = plt.figure()
        row = 5
        column = 4
        gray = None
        for fname in images:
            i += 1
            if plot_figures == True:
                fig.add_subplot(row, column, i)

            print('Loading calibration images at ' + str(int(100 * i/(len(images)))) + ' %...')
            img = cv2.imread(fname)
            if plot_figures == True:
                plt.imshow(img)
                        
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx,self.ny),None)
        
            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
        
                #Draw and display the corners
                #img = cv2.drawChessboardCorners(img, (self.nx,self.ny), corners, ret)
                #cv2.imshow('img',img)
                #cv2.waitKey(200) 

        # Do camera calibration given object points and image points, by getting the matrix coefficients
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        
        if plot_figures == True:
            plt.show()

        #cv2.waitKey(1)
        #cv2.destroyAllWindows()

    # Given an image return it's undistorted version
    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def sobel_threshold(self, gray_img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        if orient == 'x':
            sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        
        # Apply threshold
        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary

    def mag_threshold(self, gray_image, sobel_kernel=3, thresh=(0, 255)):
        # Calculate gradient magnitude
        # Apply threshold
        sobelx = self.sobel_threshold(gray_image, 'x', sobel_kernel)
        sobely = self.sobel_threshold(gray_image, 'y', sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    def dir_threshold(self, gray_image, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate gradient magnitude
        # Apply threshold
        sobelx = self.sobel_threshold(gray_image, 'x', sobel_kernel)
        sobely = self.sobel_threshold(gray_image, 'y', sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    # Define a function that takes an image and returns the combined color and gradient
    # thresholds and the staked image
    def binary_from_combined_thresholds(self, img, sobel_kernel = 3, thresh_color_s_channel = (0, 255), 
            thresh_sobel_x = (20, 100), thresh_dir_gradient = (0, np.pi/2), thresh_magnitude = (0, 255)):
    # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]

        # Grayscale image
        # NOTE: we already saw that standard grayscaling lost color information for the lane lines
        # Explore gradients in other colors spaces / color channels to see what might work better
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Sobel x & threshold x gradient
        sxbinary = self.sobel_threshold(gray, 'x', sobel_kernel, thresh_sobel_x)

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh_color_s_channel[0]) & (s_channel <= thresh_color_s_channel[1])] = 1

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        # gradient Direction 
        dir_binary = self.dir_threshold(gray, sobel_kernel, thresh_dir_gradient)
        mag_binary = self.mag_threshold(gray, sobel_kernel, thresh_magnitude)

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1) | ((dir_binary == 1) & (mag_binary == 1))] = 1

        return combined_binary, color_binary
        
    # Define a function that takes an image, number of x and y points, 
    # camera matrix and distortion coefficients and return the birdeye-view
    def corners_unwarp(self, img, corners_source, corners_destination):
        # Use the OpenCV undistort() function to remove distortion
        # undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        undist = self.undistort(img)
        img_size = (img.shape[1], img.shape[0])
        self.perspectiveM = cv2.getPerspectiveTransform(corners_source, corners_destination)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, self.perspectiveM, img_size)
        # Return the resulting image and matrix
        return warped, self.perspectiveM
        #return undist, img_size