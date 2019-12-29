## Writeup Template
Here's a [link to my video result](./data/test_videos/project_video_48cfu_final.mp4)
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./data/test_images/screenshots/calibrated_comparison.jpg "Undistorted"
[roadimage1]: ./data/test_images/screenshots/comparision_undistorted_1.jpg "Road Transformed"
[roadimage2]: ./data/test_images/screenshots/comparision_undistorted_2.jpg "Road Transformed"
[roadimage3]: ./data/test_images/screenshots/comparision_undistorted_3.jpg "Road Transformed"
[binaryimage1]: ./data/test_images/screenshots/comparision_thresholds_1.jpg "Binary Example"
[binaryimage2]: ./data/test_images/screenshots/comparision_thresholds_2.jpg "Binary Example"
[binaryimage3]: ./data/test_images/screenshots/comparision_thresholds_3.jpg "Binary Example"

[warpedimage1]: ./data/test_images/screenshots/comparision_warped_1.jpg "Warp Example"
[warpedimage2]: ./data/test_images/screenshots/comparision_warped_2.jpg "Warp Example"
[warpedimage3]: ./data/test_images/screenshots/comparision_warped_3.jpg "Warp Example"
[warpedimage4]: ./data/test_images/screenshots/comparision_warped_4.jpg "Warp Example"

[finalimage1]: ./data/test_images/saved_from_algorithm/final_lane_projection1.jpg "Final view with identified lanes"
[finalimage2]: ./data/test_images/saved_from_algorithm/final_lane_projection2.jpg "Final view with identified lanes"
[finalimage3]: ./data/test_images/saved_from_algorithm/final_lane_projection3.jpg "Final view with identified lanes"
[finalimage4]: ./data/test_images/saved_from_algorithm/final_lane_projection4.jpg "Final view with identified lanes"
[finalimage5]: ./data/test_images/saved_from_algorithm/final_lane_projection5.jpg "Final view with identified lanes"
[finalimage6]: ./data/test_images/saved_from_algorithm/final_lane_projection6.jpg "Final view with identified lanes"
[finalimage7]: ./data/test_images/saved_from_algorithm/final_lane_projection7.jpg "Final view with identified lanes"
[finalimage8]: ./data/test_images/saved_from_algorithm/final_lane_projection8.jpg "Final view with identified lanes"

[video1]: ./data/test_videos/project_video_48cfu_final.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file `./source/main_images.py` which makes use of the classes Camera and LaneDetection I wrote in `./source/camera.py` and `./source/lane_detection.py`, respectively.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the funtions `cv2.calibrateCamera()` by calling `Camera -> calibrate()`.  I applied this distortion correction to the test image using the `Camera -> undistort()` function and obtained this result: 
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][roadimage1]
![alt text][roadimage2]
![alt text][roadimage3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #70 through #159 in `camera.py`, Sobel, Color gradient, S and L channels) and lines #93 through #97 in `lane_detection.py`.   Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][binaryimage1]
![alt text][binaryimage2]
![alt text][binaryimage3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `Camera -> corners_unwarp()`, which appears in lines #160 through #167 in the file `camera.py` .  The `corners_unwarp()` function takes as inputs an image (`img_undistorted`), as well as source (`corners_source`) and destination (`corners_destination`) points.  I chose the hardcode the source and destination points in the following manner:

```python
offset1 = 350
offset2 = 520
big_x = 1230
small_x = 250

corners_source = np.float32([[small_x, 720], [small_x + offset1, 450], [big_x - offset2, 450], [big_x, 720]])
corners_destination = np.float32([[small_x, 720], [small_x, 0], [big_x, 0], [big_x, 720]])

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 240, 720      | 250, 720        | 
| 600, 450      | 250, 0      |
| 710, 450     | 1230, 0      |
| 1230, 720      | 1230, 720        |

I verified that my perspective transform was working as expected by drawing the `corners_source` and `corners_destination` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warpedimage1]
![alt text][warpedimage2]
![alt text][warpedimage3]
![alt text][warpedimage4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

For the curvature, I did it in lines #60 through #80 in my code in `lane_detection.py` in the function `LaneDetection -> measure_curvature_real()`. For the position of the vehicle, I assumed the camera is mounted at the center of the car and the deviation of the midpoint of the lane from the center of the image is the offset I'm looking for. As with the polynomial fitting, convert from pixels to meters. This in done in the function `LaneDetection -> process_image()`  implemented `lane_detection.py`, lines #259 through #263

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #60 through #80 in my code in `lane_detection.py` in the function `LaneDetection -> process_image()`.  Here is an example of my result on a test image:

![alt text][finalimage1]
![alt text][finalimage2]
![alt text][finalimage3]
![alt text][finalimage4]
![alt text][finalimage5]
![alt text][finalimage6]
![alt text][finalimage7]
![alt text][finalimage8]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./data/test_videos/project_video_48cfu_final.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- The definition of the necessary source-destination points necessary to perform the perspective transformation is hardcoded. Tuning them is very time-consuming and most problably not robust. A different setting, and camera, will require further tuning.

- To make the identified lane curves robust to noises, I've implemented a low pass filter (moving average) to find the best polynomial fit, for the current fit. This is defined in the file `./source/line.py` and by the funtion `Line -> low_pass_filter(window_size)`. After several testThe windows size was set to 11. 
- Another low pass filter was implemented for the radius of curvature and the relative position. This is implemented in `./source/line.py` by the functions `Line -> get_curvature_LPF(window_size)` and `Line -> get_relative_position_LPF(window_size)`. The windows size was set to 15 and 30, respectively.
- We try to recognize when the road is straight by printing "Straight line" to the frame. As shown in the video, although a low pass filter is implemented and a hard coded (Lines #236 to #239 in `Line -> low_pass_filter(window_size)`) handling of outliers, the text still blinks sometimes.
- A probabilistic handling (not implemented) of outliers may be a more robust way of handling noises (for example RANSAC)
- Additional preprocessing is necessary in order for the current algorithm to be able to work for the challenge video. At the moment t fails in the shadow section under the bridge.