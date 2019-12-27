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

def measure_curvature_real(ploty, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    # ploty, left_fit_cr, right_fit_cr = generate_data(ym_per_pix, xm_per_pix)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

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
    
    #combined_binary = region_of_interest(combined_binary, vertices)
    fig_handle = fig.add_subplot(8, 3, 3 * i + 3)
    plt.imshow(combined_binary)
    
    # save to file
    cv2.imwrite('../data/test_images/saved_from_algorithm/test' + str(i + 1) + '.jpg', combined_binary)
    print('Saved intermediate ../data/test_images/saved_from_algorithm/test' + str(i + 1) + '.jpg')
 
    #fig_handle.set_title('Combined S channel and  gradient thresholds')

    '''
    # Unwarp and identify lane pixels, fit with polynomial and calculate curvature
    '''
    fig3 = plt.figure(3)
    
    fig_handle = fig3.add_subplot(8, 3, 3 * i + 1)
    
    offset1 = 350
    offset2 = 520
    
    big_x = 1230
    small_x = 250
    cp_img = np.copy(img)
    cp_img = cv2.line(cp_img, (small_x, 720), (small_x + offset1, 450), [255, 0, 0], 2)
    cp_img = cv2.line(cp_img, (small_x + offset1, 450), (big_x - offset2, 450), [255, 0, 0], 2)
    cp_img = cv2.line(cp_img, (big_x - offset2, 450), (big_x, 720), [255, 0, 0], 2)
    
    corners_source = np.float32([[small_x, 720], [small_x + offset1, 450], [big_x - offset2, 450], [big_x, 720]])
    corners_destination = np.float32([[small_x, 720], [small_x, 0], [big_x, 0], [big_x, 720]])
    # plot original and warped view
    plt.imshow(cp_img)
    fig_handle = fig3.add_subplot(8, 3, 3 * i + 2)
    warped_img, M = cam.corners_unwarp(cp_img, corners_source, corners_destination)
    plt.imshow(warped_img)
 

    '''
    Binary warped image
    '''
    binary_warped, img_size = cam.corners_unwarp(combined_binary, corners_source, corners_destination)
    
    fig_handle = fig3.add_subplot(8, 3, 3 * i + 3)
    plt.imshow(binary_warped)

    
    line_left = Line()
    line_right = Line()
    
    left_detected, right_detected, leftx, lefty, rightx, righty, ploty, left_fit, right_fit, left_fitx, right_fitx, out_img = cam.fit_polynomial(
        binary_warped
    )
    #if line detected or not
    line_left.detected = left_detected
    line_right.detected = right_detected
    
    #polynomial coefficients current fit
    line_left.current_fit = left_fit
    line_right.current_fit = right_fit

    fig4 = plt.figure(4)
    

    fig_handle = fig4.add_subplot(8, 3, 3 * i + 1)
    plt.imshow(binary_warped)
    fig_handle = fig4.add_subplot(8, 3, 3 * i + 2)
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    '''
    binary_warped = out_img
    blank = np.zeros_like(binary_warped) 
    out_warped = np.dstack((blank, blank, blank))
    out_warped[lefty, leftx] = [255, 0, 0]
    out_warped[righty, rightx] = [0, 0, 255]
    '''
    '''
    Project back to original pic
    '''
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    #print(warp_zero.shape)
    text_warp = np.dstack((warp_zero.T, warp_zero.T, warp_zero.T))
    position = ((int) (text_warp.shape[1]/2 - 268/2), (int) (text_warp.shape[0]/2 - 36/2))
    position = ((int) (text_warp.shape[1]/2 - 268/2), 0)
    position = (10, 700)
    #print(position)
    text_warp = cv2.putText(
        text_warp, #numpy array on which text is written
        "48cfu", #text
        position, #position at which writing has to start
        cv2.FONT_HERSHEY_COMPLEX_SMALL, #font family
        4, #font scale
        (255, 255, 255), #font color
        10 #thickness
        ) 
    text_warp = text_warp.get()
    text_warp = np.rot90(text_warp)
    #print(result.shape)
    #print(text_warp.shape)
    color_warp = cv2.addWeighted(color_warp, 1, text_warp, 0.999, 0)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(corners_destination, corners_source)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(cam.undistort(img), 1, newwarp, 0.3, 0)
    
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit a second order polynomial to pixel positions in each fake lane line
    ##### TO-DO: Fit new polynomials to x,y in world space #####
    ##### Utilize `ym_per_pix` & `xm_per_pix` here #####
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    left_curverad, right_curverad = measure_curvature_real(ploty, left_fit_cr, right_fit_cr)
    print(left_curverad, 'm', right_curverad, 'm')

    # Write curvatures to image
    result = cv2.putText(
        result, #numpy array on which text is written
        'Left curvature: ' + str(int(left_curverad)) + 'm', #text
        (10, 30), #position at which writing has to start
        cv2.FONT_HERSHEY_COMPLEX_SMALL, #font family
        1, #font scale
        (255, 255, 255), #font color
        2#thickness
    ) 

    result = cv2.putText(
        result, #numpy array on which text is written
        'Right curvature: ' + str(int(right_curverad)) + 'm', #text
        (10, 60), #position at which writing has to start
        cv2.FONT_HERSHEY_COMPLEX_SMALL, #font family
        1, #font scale
        (255, 255, 255), #font color
        2#thickness
    ) 


    '''
    POsition vehicle with respect to center
    '''
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    xmin = 0
    xmax = result.shape[1]
    lane_left = left_fitx[0]
    lane_right = right_fitx[0]
    offset = (xmax - xmin) / 2 - (lane_right - lane_left)/2

    relative_position = offset * xm_per_pix
    result = cv2.putText(
            result, #numpy array on which text is written
            'Vehicle relative position: ' + str((float("{0:.2f}".format(relative_position)))) + 'm', #text
            (10, 90), #position at which writing has to start
            cv2.FONT_HERSHEY_COMPLEX_SMALL, #font family
            1, #font scale
            (255, 255, 255), #font color
            2#thickness
        ) 


    # Show and save result
    fig_handle = fig4.add_subplot(8, 3, 3 * i + 3)
    cv2.imwrite('../data/test_images/saved_from_algorithm/final_lane_projection' + str(i + 1) + '.jpg',  cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print('Saved intermediate ../data/test_images/saved_from_algorithm/final_lane_projection' + str(i + 1) + '.jpg')
    plt.imshow(result)

    i += 1
    #if i >= 1:
    #    break


plt.show(block = False)



input("Press Enter to continue...")