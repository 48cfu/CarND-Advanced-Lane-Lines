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
from camera import Camera
from line import Line

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


'''
Compute the camera calibration matrix and distortion coefficients
given a set of chessboard images.
'''
## Instantiate camera and and calibrate it
cam = Camera()
cam.calibrate()
fname = '../data/camera_cal/calibration1.jpg'
example_img = mpimg.imread(fname)  

''' 
First rubric point: 
Provide an example of a distortion-corrected image
'''
fig = plt.figure()
#  load original image and display it
undistorted_example_img = cam.undistort(example_img)
fig_handle = fig.add_subplot(1, 2, 1)
plt.imshow(example_img)
fig_handle.set_title('Original image')
plt.show(block = False)

# undistort the image and display it to the same figure
fig_handle =fig.add_subplot(1, 2, 2)
plt.imshow(undistorted_example_img)
fig_handle.set_title('Undistorted image')
plt.show(block = False)


'''
Provide an example of a binary image result.
'''
# Use graphical interface to pick
# Load test images
images = glob.glob('../data/test_images/test*.jpg')

sobel_kernel = 9
thresh_color_s_channel = (90, 255) # from Gradients and Color Spaces: HLS Quiz
thresh_sobel_x = (20, 100)
thresh_dir_gradient = (0.7, 1.3)
thresh_magnitude = (30, 100)
'''
230 max_y
1240 max_y

600 450
840 450
'''
left_low = (100, 720)
left_high = (600, 450)
right_low = (1240, 720)
right_high = (805, 450)
vertices = np.array([[left_low, left_high, right_high, right_low]], dtype=np.int32)


i = 0
for fname in images:
    fig = plt.figure(2)
    img = mpimg.imread(fname) #in RGB
    '''
    Get combined binary image
    '''
    combined_binary, color_binary = cam.binary_from_combined_thresholds(
        img, sobel_kernel, thresh_color_s_channel, thresh_sobel_x, 
        thresh_dir_gradient, thresh_magnitude)

    # Plotting thresholded images
    fig_handle = fig.add_subplot(8, 3, 3 * i + 1)
    plt.imshow(img)
    #fig_handle.set_title('Original image')

    fig_handle = fig.add_subplot(8, 3, 3 * i + 2)
    plt.imshow(color_binary)
    #fig_handle.set_title('Stacked thresholds')
    
    combined_binary = region_of_interest(combined_binary,vertices)
    fig_handle = fig.add_subplot(8, 3, 3 * i + 3)
    plt.imshow(combined_binary)
    
    # save to file
    cv2.imwrite('../data/test_images/saved_from_algorithm/test' + str(i + 1) + '.jpg', combined_binary)
    print('Saved in ../data/test_images/saved_from_algorithm/test' + str(i + 1) + '.jpg')
 
    #fig_handle.set_title('Combined S channel and  gradient thresholds')

    '''
    # Unwarp and identify lane pixels, fit with polynomial and calculate curvature
    '''
    fig3 = plt.figure(3)
    
    fig_handle = fig3.add_subplot(8, 2, 2 * i + 1)
    
    offset1 = 330
    offset2 = 450
    
    cp_img = np.copy(img)
    cp_img = cv2.line(cp_img, (250, 720), (250 + offset1, 450), [255, 0, 0], 2)
    cp_img = cv2.line(cp_img, (250 + offset1, 450), (1150 - offset2, 450), [255, 0, 0], 2)
    cp_img = cv2.line(cp_img, (1150 - offset2, 450), (1150, 720), [255, 0, 0], 2)
    
    corners_source = np.float32([[255, 720], [255 + offset1, 450], [1150 - offset2, 450], [1150, 720]])
    corners_destination = np.float32([[250, 720], [250, 0], [1150, 0], [1150, 720]])
    # plot original and warped view
    plt.imshow(cp_img)
    fig_handle = fig3.add_subplot(8, 2, 2 * i + 2)
    warped_img, M = cam.corners_unwarp(cp_img, corners_source, corners_destination)
    plt.imshow(warped_img)
    i += 1

    '''
    Binary warped image
    '''
    binary_warped = cam.corners_unwarp(combined_binary, corners_source, corners_destination)

plt.show(block = False)



input("Press Enter to continue...")