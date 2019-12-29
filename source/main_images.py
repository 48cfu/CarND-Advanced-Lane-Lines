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
import matplotlib.image as mpimg

#%matplotlib qt
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')
from lane_detection import LaneDetection

'''
Compute the camera calibration matrix and distortion coefficients
given a set of chessboard images.
'''
## Instantiate ride
ride = LaneDetection()

# Get images path
images = glob.glob('../data/test_images/test*.jpg')
i = 0
for fname in images:
    original_img = mpimg.imread(fname) #in RGB
    undistorted_img = ride.camera.undistort(original_img)
    
    '''
    Get combined binary image
    '''
    combined_binary = ride.thresholding_pipeline(undistorted_img)

    fig = plt.figure(i * 4)
    # Plotting thresholded images
    fig_handle = fig.add_subplot(1, 2, 1)
    plt.imshow(original_img)
    fig_handle.set_title('Original image')

    fig_handle = fig.add_subplot(1, 2, 2)
    plt.imshow(combined_binary, cmap='gray')
    fig_handle.set_title('Combined S channel and  gradient thresholds')

    '''
    # Unwarp and identify lane pixels, fit with polynomial and calculate curvature
    '''
    offset1 = 350
    offset2 = 520
    big_x = 1230
    small_x = 250
    cp_img = np.copy(undistorted_img)
    cp_img = cv2.line(cp_img, (small_x, 720), (small_x + offset1, 450), [255, 0, 0], 2)
    cp_img = cv2.line(cp_img, (small_x + offset1, 450), (big_x - offset2, 450), [255, 0, 0], 2)
    cp_img = cv2.line(cp_img, (big_x - offset2, 450), (big_x, 720), [255, 0, 0], 2)
    corners_source = np.float32([[small_x, 720], [small_x + offset1, 450], [big_x - offset2, 450], [big_x, 720]])
    corners_destination = np.float32([[small_x, 720], [small_x, 0], [big_x, 0], [big_x, 720]])
    
    # plot original and warped view
    fig3 = plt.figure(i * 4 + 1)
    fig_handle = fig3.add_subplot(1, 3, 1)
    plt.imshow(cp_img)
    fig_handle.set_title('Source points')

    fig_handle = fig3.add_subplot(1, 3, 2)
    warped_img, M = ride.camera.corners_unwarp(cp_img, corners_source, corners_destination)
    fig_handle.set_title('Destination points')
    plt.imshow(warped_img)
 

    '''
    Binary warped image
    '''
    fig_handle = fig3.add_subplot(1, 3, 3)
    binary_warped, M = ride.camera.corners_unwarp(combined_binary, corners_source, corners_destination)
    fig_handle.set_title('Perspective transform of Binary image')
    plt.imshow(binary_warped, cmap='gray')

    
    final_result = ride.process_image(undistorted_img)
    fig = plt.figure(i * 4 + 2)
    # Plotting thresholded images
    fig_handle = fig.add_subplot(1, 2, 1)
    plt.imshow(original_img)
    fig_handle.set_title('Original image')

    fig_handle = fig.add_subplot(1, 2, 2)
    cv2.imwrite('../data/test_images/saved_from_algorithm/final_lane_projection' + str(i + 1) + '.jpg',  cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
    print('Saved Final ../data/test_images/saved_from_algorithm/final_lane_projection' + str(i + 1) + '.jpg')

    plt.imshow(final_result)
    fig_handle.set_title('Identified lanes and curvature')

    fig = plt.figure(i * 4 + 3)
    # Plotting thresholded images
    fig_handle = fig.add_subplot(1, 2, 1)
    plt.imshow(original_img)
    fig_handle.set_title('Original image')

    fig_handle = fig.add_subplot(1, 2, 2)
    plt.imshow(undistorted_img)
    fig_handle.set_title('Undistorted image')

    i += 1


plt.show(block = False)


fname = '../data/camera_cal/calibration1.jpg'
example_img = mpimg.imread(fname)  


''' 
First rubric point: 
Provide an example of a distortion-corrected image
'''
fig = plt.figure(100)
#  load original image and display it
undistorted_example_img = ride.camera.undistort(example_img)
fig_handle = fig.add_subplot(1, 2, 1)
plt.imshow(example_img)
fig_handle.set_title('Original image')
plt.show(block = False)

# undistort the image and display it to the same figure
fig_handle =fig.add_subplot(1, 2, 2)
plt.imshow(undistorted_example_img)
fig_handle.set_title('Undistorted image')
plt.show(block = False)

input("Press Enter to continue...")