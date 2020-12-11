[//]: # (Image References)

[undist_chessboard]: ./writeup/undist_chessboard.png "Undistorted and warped chessboard"
[distortion_corrected]: ./writeup/distortion_corrected.png "Original image and distortion-corrected image"
[perspective_transform]: ./writeup/perspective_transform.png "Perspective transform to birds-eye view"
[threshold_image]: ./writeup/threshold_image.png "Color and gradient threshold"
[color_gradient]: ./writeup/color_gradient.png "Threshold is not robust in third image"
[radial_distortion]: ./writeup/calibration_radial_distortion.png "Radial distortion"
[tangential_distortion]: ./writeup/calibration_tangential_distortion.png "Tangential distortion"
[lane_detection_process]: ./writeup/process.png "Lane detection process"
[histogram]: ./writeup/histogram.png "Histogram concept"
[sliding_window]: ./writeup/sliding_window.png "Sliding window and polynomial"
[best_weight]: ./writeup/best_weight.png "Compute best fit"
[color_fit_lines]: ./writeup/color_fit_lines.jpg "Fit Visual"
[project_video_gif]: ./writeup/project_video_20_25.gif "project_video_gif"
[challenge_video_gif]: ./writeup/challenge_video_00_04.gif "challenge_video_gif"
[issue_01]: ./writeup/issue_01.png "Too much noise"
[issue_02]: ./writeup/issue_02.png "Shadow hides the lane line"

## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

About my code
---

Thanks for reading my second project writeup. All the code snippets or the functions mentioned below can be found in the IPython notebook located in **"CarND-Advanced-Lane-Lines/advanced-lane-lines.ipynb"**. The output videos can be found in the path **"CarND-Advanced-Lane-Lines/output_videos"**. Let's start!

### Camera Calibration

Before we start to detect lane lines from image, we need to fix the image distortion problem first. Below two images are coming from [MathWorks document](https://de.mathworks.com/help/vision/ug/camera-calibration.html), they briefly describe the two types of distortion.

#### Radial distortion
![Radial distortion][radial_distortion]

#### Tangential distortion
![Tangential distortion][tangential_distortion]

To correct the distorted image, I need to compute camera matrix and distortion coefficients. The code for this step is contained in the function `camera_calibration()`. In this function, I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function and save them as the variable `mtx` and `dist`. Then, using `cv2.undistort()` to correct image.
 
I applied this distortion correction to the image using the `undistort()` function and obtained this result:

![Undistorted and warped chessboard][undist_chessboard]

### Pipeline (single image)

#### 1. Get the distortion-corrected image.

As describing in the previous part, I calibrate the image by using `undistort()` function. Here is one result from test image.

![Original image and distortion-corrected image][distortion_corrected]

#### 2. Perspective transform

In this step, I will transform the input image into the birds-eye view so that the lane lines are (more or less) parallel in the transformed image:

![birds-eye view][perspective_transform]

I define my `src` and `dst`, and then use `cv2.getPerspectiveTransform` to compute the transform matrix `perspective_mat` and its inverse matrix `perspective_mat_inv` for later usage. This computation is done in the function `get_perspective_matrix()`. After I have the transform matrix, I use `cv2.warpPerspective` to transform the image, this part is wrapped in the function `perspective_transform()`. Here is my source and destination points:

|    Corner    |  Source   | Destination |
| :----------: | :-------: | :---------: |
|   Top left   | 400, 500  |   0, 300    |
| Bottom left  | 200, 720  |  300, 720   |
|  Top right   | 880, 500  |  1200, 300  |
| Bottom right | 1080, 720 |  900, 720   |

I also use the above image as referece (two horizontal red lines at top left part and two vertical red lines at the middle bottom part of image) to compute the values `ym_per_pix` and `xm_per_pix`. Which are used to convert pixels to real-world meter measurements. I define them directly in line 29 and 30 of the code `class Line`.

```python
class Line():
    def __init__(self):
        ...
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3 / 180  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 610  # meters per pixel in x dimension
```

#### 3. Thresholded binary image

The function related to gradient threshold are `abs_sobel_thresh()`, `mag_threshold()` and `dir_threshold()`. And the function related to color threshold are `hls_select()` and `luv_select()`. 

For yellow lane line, I use HLS color space to extract yellow color. For white lane line, I use LUV color space to extract white color. Then combining with sobel x gradient, magnitude and direction of gradient to generated the threshold binary image.

I currently can't find more robust threshold parameters. Compare the following three images extracted from the video `project_video.mp4`, `challenge_video.mp4` and `harder_challenge_video.mp4`. The first two looks pretty good, but the third one has too much noise, and they will seriously impact the following lane detection process.

![Compare the threshold in different environments][color_gradient]

#### 4. Using sliding window to identify lane-line pixels and fit them with a polynomial

The reader can find the code of this part in the function `find_lane_pixels()`.

I take a histogram along all the columns in the lower half of the image, and choose the highest peak from `x=0` to `x=binary_image.shape[0]//2` as the first sliding window's center `x` for left line and from `x=binary_image.shape[0]//2` to `x=binary_image.shape[0]` for right part.

![The concept of histogram][histogram]

After I have the center `x` for left and right sliding windows, I compute the boundary of each sliding window according to `window_height` and `margin` (From line 72 to 82). Then I seach the lane pixels in each window (From line 85 to 88). And update center `x` for next sliding window if the number of pixels found in current window are more than the threshold `minpix` (From line 112 to 115).

![The sliding window process][sliding_window]

In the function `find_lane_pixels()`, it takes `binary_warped`, `left_line` and `right_line` as parmeters. `binary_warped` is the image output from previous step. `left_line` and `right_line` are the instance of `class Line()` which are used to record the previous lane information. This function will use two different ways to detect lane line pixels accroading to `class Line()` member `detected` (From line 54 to 70 and 112 to 115).

1. If lane line is not detected in previous frame, it will use sliding window to search from scratch.
2. Otherwise, it will use `class Line()` member `best_fit` to compute each search window's x center, and then search lane pixels just like sliding windows.

After I have lane pixels, I use `np.polyfit()` to compute `class Line()`'s member `current_fit`(current polynomial) and then update `best_fit`(best polynomial). You can find this part in `class Line()`, from line 72 to 79. The rule of update `best_fit` is using the weight. This weight is equal to `empty windows / total windows`. I use below image as example:

![Compute the weight of best_fit][best_weight]

You can see that there are two windows do not find any lane pixels on left side, and four windows on right side. So that left lane's `best_fit` will be updated as `(2/9) * best_fit + (7/9) * current_fit`, and right lane's `best_fit` will be `(4/9) * best_fit + (5/9) * current_fit`. 

OK, that's all. Here is a step by step image to show the full process of my lane line detection:

![The process of lane line detection][lane_detection_process]

#### 5. The radius of curvature of the lane and the position of the vehicle with respect to center

![Curvature][color_fit_lines]

$$ R_{curve} = \frac{[1+(\frac{dx}{dy})^2]^{3/2}}{\left | \frac{d^2x}{dy} \right |} = \frac{[1+(2Ay+B)^2]^{3/2}}{\left | 2A \right |} $$

I compute the radius of curvature accroading to the above formula. The related code is written in `class Line()`'s member function `measure_curvature_real()`. The shift related to center is computed in the function `process_image()`, line 14 to 16.

#### 6. Plot the lane area back down onto the road

I implemented this step in lines 18 through 72 in my code `process_image()`. Here are two gif images showing the part of result.

![project_video][project_video_gif]

![challenge_video][challenge_video_gif]

---

### Pipeline (video)

Here are the link of my output videos:
1. [project_video.mp4 output](../output_videos/project_video.mp4)
2. [challenge_video.mp4 output](../output_videos/challenge_video.mp4)

---

### Discussion

1. Color and gradient threshold part is not good enough. As I mentioned previously, the parameters I currently use is not robust enough for the `harder_challenge_video.mp4` case. It has too much noise to impact the lane detection. You can see below image as example. The noise on right side let histogram step dicides a wrong begining sliding window. My idea of how to get better threshold parameters is using some black-box hyperparameter tuning techniques, for example, [CMA-ES](https://en.wikipedia.org/wiki/CMA-ES). But I can't find the true label training data. So I skip this idea currently.

![Noise problem][issue_01]

2. Color threshold is very easily impacted by the real environment. For example, in the below image, the shadow let both yellow and white lane undetectable. Needless to say, in the rainy night, there will have so many environment objects be reflected on the road by the rain. I currently have no idea about how to improve this terrible case.

![Shadow problem][issue_02]

---

### Reference
1. A very amazing result: [Lane Tracker](https://github.com/pierluigiferrari/lane_tracker)
2. Deep learning method: [LaneNet-Lane-Detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection)

