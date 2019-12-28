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

import tkinter as tk, threading
import imageio
from PIL import Image, ImageTk


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

#%matplotlib qt
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')

from lane_detection import LaneDetection


'''
Compute the camera calibration matrix and distortion coefficients
given a set of chessboard images.
'''
## Instantiate ride
ride = LaneDetection()

''' 
Video rubric
'''
# Video  file path
video_name = '../data/test_videos/project_video.mp4'
video_name_save = '../data/test_videos/project_video_48cfu.mp4'
#video = imageio.get_reader(video_name)

video = VideoFileClip(video_name)
new_clip = video.fl_image(ride.process_image)
#get_ipython().run_line_magic('matplotlib', 'time')
new_clip.write_videofile(video_name_save, audio=False)

input("Press Enter to continue...")