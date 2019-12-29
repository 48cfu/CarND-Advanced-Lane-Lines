# Define a class to represent the camera
import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt

from camera import Camera
from line import Line

class LaneDetection():
    def __init__(self):
        ''' 
        Configuration parameters for each frame: tuned in main_images.py
        '''
        self.sobel_kernel = 3
        self.thresh_color_s_channel = (80, 255) # from Gradients and Color Spaces: HLS Quiz
        self.thresh_sobel_x = (20, 100)
        self.thresh_dir_gradient = (0.7, 1.3)
        self.thresh_magnitude = (30, 100)
        self.left_low = (100, 720)
        self.left_high = (600, 450)
        self.right_low = (1240, 720)
        self.right_high = (805, 450)
        self.vertices = np.array([[self.left_low, self.left_high, self.right_high, self.right_low]], dtype=np.int32)

        #set camera
        self.camera = Camera()
        self.camera.calibrate()

        #lines
        self.line_left = Line()
        self.line_right = Line()

    def region_of_interest(self, img, vertices):
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

    def measure_curvature_real(self, ploty, left_fit_cr, right_fit_cr):
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


    def abs_sobel_thresh(self, img, orient='x', thresh_min=20, thresh_max=100):
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return binary_output

    def thresholding_pipeline(self, img_undistorted):
        combined_binary, color_binary = self.camera.binary_from_combined_thresholds(
            img_undistorted, self.sobel_kernel, self.thresh_color_s_channel, self.thresh_sobel_x, 
            self.thresh_dir_gradient, self.thresh_magnitude)
        return combined_binary
        
    def process_image(self, img_original):
        ''' 
        first undistort
        '''
        img_undistorted = self.camera.undistort(img_original)

        '''
        Get combined binary image
        '''

        combined_binary, color_binary = self.camera.binary_from_combined_thresholds(
            img_undistorted, self.sobel_kernel, self.thresh_color_s_channel, self.thresh_sobel_x, 
            self.thresh_dir_gradient, self.thresh_magnitude)
        
        '''
        # Unwarp and identify lane pixels, fit with polynomial and calculate curvature
        '''
        offset1 = 350
        offset2 = 520
        big_x = 1230
        small_x = 250

        corners_source = np.float32([[small_x, 720], [small_x + offset1, 450], [big_x - offset2, 450], [big_x, 720]])
        corners_destination = np.float32([[small_x, 720], [small_x, 0], [big_x, 0], [big_x, 720]])
        print(corners_source)
        print(corners_destination)
        cp_img = np.copy(img_undistorted)

        img_size = (img_undistorted.shape[1], img_undistorted.shape[0])
        #warped_img, M = self.camera.corners_unwarp(cp_img, corners_source, corners_destination)

        '''
        Binary warped image
        '''
        binary_warped, M = self.camera.corners_unwarp(combined_binary, corners_source, corners_destination)

        #check if left and right were detected
        if self.line_left.detected == False or self.line_right.detected == False:
            left_detected, right_detected, leftx, lefty, rightx, righty, ploty, left_fit, right_fit, left_fitx, right_fitx, out_img = self.camera.fit_polynomial(
                binary_warped
            )
        else:
            #TODO: change to search around previous one
            #left_detected, right_detected, leftx, lefty, rightx, righty, ploty, left_fit, right_fit, left_fitx, right_fitx, out_img = self.camera.search_around_poly(
            #    binary_warped, self.line_left.best_fit, self.line_right.best_fit
            #)
            left_detected, right_detected, leftx, lefty, rightx, righty, ploty, left_fit, right_fit, left_fitx, right_fitx, out_img = self.camera.fit_polynomial(
                binary_warped
            )
        #color_binary = np.dstack((combined_binary, combined_binary, combined_binary)) * 255
        #left line
        self.line_left.detected = left_detected
        self.line_left.allx = leftx #noisy lane pixels x
        self.line_left.ally = lefty #noisy lane pixels y
        self.line_left.current_fit.append(left_fit) #polinomial coefficients of current fit
        self.line_left.best_fit = self.line_left.low_pass_filter()
        
        #right line
        self.line_right.detected = right_detected
        self.line_right.allx = rightx #noisy lane pixels x
        self.line_right.ally = righty #noisy lane pixels y
        self.line_right.current_fit.append(right_fit) #polinomial coefficients of current fit
        self.line_right.best_fit = self.line_right.low_pass_filter()

        try:
            left_fitx = self.line_left.best_fit[0]*ploty**2 + self.line_left.best_fit[1]*ploty + self.line_left.best_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            #print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
        try:
            right_fitx = self.line_right.best_fit[0]*ploty**2 + self.line_right.best_fit[1]*ploty + self.line_right.best_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            #print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty


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
        position = (30, 700)
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
        
        color_warp = cv2.addWeighted(color_warp, 1, text_warp, 0.999, 0)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv = cv2.getPerspectiveTransform(corners_destination, corners_source)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img_undistorted.shape[1], img_undistorted.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(img_undistorted, 1, newwarp, 0.3, 0)
 
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
        
        left_curverad, right_curverad = self.measure_curvature_real(ploty, left_fit_cr, right_fit_cr)
        
        # remove outliers
        if left_curverad < 10000:
            self.line_left.radius_of_curvature.append(left_curverad)
        if right_curverad < 10000:
            self.line_right.radius_of_curvature.append(right_curverad)
        
        text_radius = str(int(0.5 * self.line_left.get_curvature_LPF() + 0.5 * self.line_right.get_curvature_LPF()))

        if int(0.5 * self.line_left.get_curvature_LPF() + 0.5 * self.line_right.get_curvature_LPF()) > 3000:
            text_radius = 'Straight line'

        result = cv2.putText(
            result, #numpy array on which text is written
            'Lane curvature [m]: ' + text_radius, #text
            (10, 30), #position at which writing has to start
            cv2.FONT_HERSHEY_COMPLEX_SMALL, #font family
            1, #font scale
            (255, 255, 255), #font color
            2#thickness
        ) 

        '''
        POsition vehicle with respect to center
        '''
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        mid_lane = 0.5 * (self.line_left.best_fit[0]*img_undistorted.shape[0]**2 + self.line_left.best_fit[1]*img_undistorted.shape[0] + self.line_left.best_fit[2]) + 0.5*(self.line_right.best_fit[0]*img_undistorted.shape[0]**2 + self.line_right.best_fit[1]*img_undistorted.shape[0] + self.line_right.best_fit[2])
        offset = mid_lane - 0.5 * img_undistorted.shape[1]
        relative_position = offset * xm_per_pix
        self.line_left.line_base_pos.append(relative_position)
        
        result = cv2.putText(
                result, #numpy array on which text is written
                'Vehicle relative position [m]: ' + str(round(self.line_left.get_relative_position_LPF(), 2)), #text
                (10, 60), #position at which writing has to start
                cv2.FONT_HERSHEY_COMPLEX_SMALL, #font family
                1, #font scale
                (255, 255, 255), #font color
                2#thickness
            ) 
        
        return result
        #return color_binary