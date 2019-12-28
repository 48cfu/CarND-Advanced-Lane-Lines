import numpy as np
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [] #[np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = [] 
        #distance in meters of vehicle center from the line
        self.line_base_pos = [] 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  

    #make sure to append new values to current_fit before calling
    def low_pass_filter(self, window_size = 11):
        #shape = self.current_fit.shape
        snapshot = self.current_fit[-window_size:]
 
        if snapshot[-1:] == [10, 20, 0]:
            snapshot = np.delete(snapshot, -1)
            
        best_fit = np.mean(snapshot, axis = 0)

        self.best_fit = best_fit
        return best_fit
    
    #make sure to append new values to radius_of_curvature before calling
    def get_curvature_LPF(self, window_size = 15):
        snapshot = self.radius_of_curvature[-window_size:]
        curvature = np.mean(snapshot, axis = 0)
        return curvature

    #make sure to append new values to line_base before calling
    def get_relative_position_LPF(self, window_size = 30):
        snapshot = self.line_base_pos[-window_size:]
        relative_position = np.mean(snapshot, axis = 0)
        return relative_position