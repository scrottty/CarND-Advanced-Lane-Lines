# **Advanced Lane Finding Project**

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

[chessboards]: ./images/chessboards.png "Chessboard Undistortion"
[undist]: ./images/testImage_undist.png "Image Undistortion"
[b_channel]: ./images/b_channel.png "B Channel"
[l_channel]: ./images/l_channel.png "L Channel"
[b_l_combined]: ./images/b_l_combination.png "Combination of B and L Channels"
[gradient_combined]: ./images/gradient_combination.png "Thresholded Gradient of Image"
[color_grad_combined]: ./images/color_grad_combined.png "Combination of Colorspaces and Gradient"
[perspectiveTransform]: ./images/perspectiveTransform.png "perspective Transform of Binary Image"
[histogram]: ./images/histogram.png "Histogram of Pixel Values In Binary Image"
[slidingWindow]: ./images/slidingWindow.png "Examples of Sliding Window Algorithm"
[radiusEquation]: ./images/radiusEquation.png "Equation for the Radius of Curvature"
[finalResult]: ./images/finalResult.png "Lane Overlayed with Lines and Results"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Project code
Please find the code for the project in [*workbook_video.ipynb*](https://github.com/scrottty/CarND-Advanced-Lane-Lines/blob/master/workbook_video.ipynb). This is the 'tidy' pipeline used in producing the video. For the development workbook where the **Camera Calibration** was computed please see [*workbook_images.ipynb*](https://github.com/scrottty/CarND-Advanced-Lane-Lines/blob/master/workbook_images.ipynb). Note this is really messy and was used for exploring ideas and finding thresholds.


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in section **Camera Calibration** in [*workbook_images.ipynb*](https://github.com/scrottty/CarND-Advanced-Lane-Lines/blob/master/workbook_images.ipynb).

The first step was to construct the object and image points representing the (x,y,z) coordinates of the chessboard corners in the destination and source coordinates respectively. The object points (`objpoints`) are the same for each calibration image, an array that is replicated to match the image points (`imgpoints`).

To construct the image points the open cv function `cv2.findChessboardCorners()` was run on grayscale images of a 9x6 chessboard taken from multiple angles. Two of the images did not have all of the corner visible so where not used in the calibration.

To undistort the image the open cv function `cv2.calibrateCamera()`. This produces the camera calibration matrix (`mtx`) and distortion coefficients (`dist`) which are then used in the `cv2.undistort()` function to return an undistorted result. The result of the undistortion can be seen below. The camera calibration matrix and the distortion coefficients are saved for use in the video pipeline.

![alt text][chessboards]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

As each frame is captured from the camera it must first be undistorted. This is done in the `undistortImage` function in the video pipeline. Here is an example of the undistrted image

![alt text][undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image that represented the only the lane lines from the image. The code for the colorspaces is in the `colorspace` function whilst the code for the gradients is in the `gradient` function in [*workbook_video.ipynb*](https://github.com/scrottty/CarND-Advanced-Lane-Lines/blob/master/workbook_video.ipynb)

For the colorspaces I ended up using a combination of the B channel in the LAB colorspace and the L channel in the LUV colorspace thresholded with `(160,255)` and `(210,255)` respectively. Other colorspaces, channels and thresholds were played with in [*workbook_images.ipynb*](https://github.com/scrottty/CarND-Advanced-Lane-Lines/blob/master/workbook_images.ipynb).

The B channel was choosen for it ability to pick up the yellow lines. On both the light and dark road surfaces it could easily pull the yellow lines distinctly from the other objects making it easy to threshold.

![alt text][b_channel]

The L channel was chosen as it could pull the white lines from the images well. Like other channels it did struggle to pick up the white lines on the lighter road surfaces but did the best.

![alt text][l_channel]

The S channel from the HLS colorspace was initially used as well however after running on the project video it was found to introduce too much noise at certain points and so was removed. The final combination of the B and L channel on the test images is shown below. The green points are the contribution of the B channel and the blue points are the contribution of the L channel

![alt text][b_l_combined]

The implementation of the colorspace is below:

```py
    def colorspace(image):
        B = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:,:,2]
        L = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)[:,:,0]
        B = createBinary(B, (160, 255))
        L = createBinary(L, (210, 255))
        binaryCombined = np.zeros_like(B)
        binaryCombined[(B==1) | (L==1)] = 1
        return binaryCombined
```

For the gradient there was the choice of absolute value in both x and y, magnitude and direction of gradient. Initially they were all chosen but later the magnitude and gradient direction were removed as they brought in noise on the lighter section of the road. The gradient was found to be secondary to the colorspace but could at times find lines in tricky images where the colorspace could not. The absolute value of the gradient was the value that found these tricky areas so was kept. The values were thresholded as in the table below and were combined only when found in both the x and y gradient.

|Value|Threshold|Sobel Kernel|
| --- | --- | :---: |
| Gradient X | `(50,200)` | 15|
| Gradient Y | `(20,200)` | 15|

The results of the gradients are shown below

![alt text][gradient_combined]

In the final implementation of the pipeline I included videos for both with and without the gradient detecting lane lines. I found that the gradient often introduced unwanted noise and for the most part made the solution worse especially in the calculation of the radius of curvature. The result without a gradient was definately cleaner and with the exception of the final shadow that coincides with a change in tarmac colour performs well. At this final point it does predict a and incorrect curvature but this could be fixed by ignoring the bad value effecting it.

The implimentation of the gradient detection is below:
```py
    def gradient(image):
        grad_absX = gradient_abs(image, orient='x', sobel_kernel=15, thresh=(50,200))
        grad_absY = gradient_abs(image, orient='y', sobel_kernel=15, thresh=(20,200))

        binaryCombined = np.zeros_like(grad_absX)
        binaryCombined[((grad_absX==1)&(grad_absY==1))] = 1
        return binaryCombined
```

The combination of the colorspaces and gradient produced the following result on the test images:

![alt text][color_grad_combined]

The image was also masked to the area of interest similar to project 1 to stop any accidental noise from the surroundings distracting the pipeline. This is in the function `preprocessingImage()`

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform in the video pipeline is in the function `perspective`, shown below
```py
    def perspective(image):
        return cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
```

This uses the `M` matrix calculated in [*workbook_images.ipynb*](https://github.com/scrottty/CarND-Advanced-Lane-Lines/blob/master/workbook_images.ipynb). This was calculated by the openCV function `cv2.getPerspectiveTransform` which takes in source and destination matricies for 4 points on the image.

The source points were chosen by eye using an image of a straight road. They were adjusted to best make the resulting lines, in the transformed image, vertical and parallel (or the best i could get it). They ended up as:
```py
    src = np.float32([[580,460],
                      [700,460],
                      [1040,imshape[0]-40],
                      [260, imshape[0]-40]])
    dst = np.float32([[350,0],
                      [930,0],
                      [930,imshape[0]],
                      [350, imshape[0]]])
```

The `M` matrix from the function was then saved to file for use in the video pipeline. There it is used by the `cv2.warpPerspective()` to change the perspective of the image for line finding. The images below show the perspective transform

![alt text][perspectiveTransform]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To detect which pixel in the transformed images belonged to each lane a sliding windown search done. The steps for this are as follows:
1. Take a histogram of the bottom half of the image to detect where each lane line starts. The histogram, being a summation of the pixels, will have a high value at the position about which the pixels a collected. This finds the starting center of the first window. The left lane is assumed as the peak on the left hand side of the image and the right lane is assumed as the peak on the right hand side of the image. Example histogram is shown below:

![alt text][histogram]

2. Place a window with a set size, defined by a margin about the found center point and a height, on the image and 'collect' all of the pixels within the window and assign it to the lane.
3. Next 'slide' the window up on window position and again collect the pixels within the window for that lane. If a suitable number of pixels are found within the window recenter the window for the next 'slide'
4. Fit a polynomial to the collected points for each lane using `np.polyfit` with a order of 2. This fitted polynomial represents the lane line, assuming that the points found are an even spread of pixels representing the lane line.

The code for this is in the `findLanesWindowSlide()` function. This produces the following result:

![alt text][slidingWindow]

To speed the process up when running the video pipeline the window sliding section was removed and pixels are searched for within a margin about the previously line, given it was an acceptable result. This code can be found in the `findLanesLocal()` function.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature was calculated in the 'radiusOfCurvature()' function along with the top and bottom positions of the fitted lane line.

The radius of curvature was calculated using the following equations:

![alt text][radiusEquation]

To convert it from pixel space to real space the values were multiplied by the following conversion factors:
```py
    y_meters_per_pixel = 30/720 #30 meters for the height of the transformed image
    x_meters_per_pixel = 3.7/580 #3.7 meters between the lane lines that are assumed as by the American Road Standard
```

The the bottom position of the lane in the image was used to calcuate the position of the car in the lane. The assumption of the middle of the car being in the middle of the image meant the mid-point between the two lane lines could be compared to the center of the image to calcuate the distance from the center of the lane. The code is in the `filterLanes()` function as is:

```py
    centerOffset = (rightLane.basePos - leftLane.basePos) - (640*x_meters_per_pixel) # 640 being 1280/2
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this in the `drawLanes()` function in the video pipeline. The averaged fitted lane lines are used to form a polygon which is then overlayed to the image to show the lane lines and driving area. The radius of curvature and the distance the car is from the middle of the lane are printed on the top left of the image. The binary image, upon which the lane finding is done, is placed in the top right corner for reference.

![alt text][finalResult]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here links to my final videos for results [with gradient](./videos/project_video_gradient.mp4) and [without gradient](./videos/proejct_video_nogradient.mp4)

As mentioned above int he gradient sections i found the gradient introduced a lot of noise and this caused worse performance for the most of the video. It helped in the final shadow and change of tarmac colour but other than that the colorspace thresholding worked suitable. It would be interesting to hear how to better handle the gradient to stop it from introducing so much noise!

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Though the final solution meets the criteria there a quite a few problems with my current implimentation:
1. The 'lane finding model' is very overfit to this example. When tried on the harder videos it falls over completely. This is because the thresholds to find the lanes have been manually tuned and adjusted upon the test and problem frames to produce the best result for this video. To fix this I imagine that a machine learning approach would be better with more training data. This way a large amount of training data could be passed to the model to let it become more generalised and stop it fitting purely to this case.
2. Shadows, changes in the brightness of the image and changes in the color of the tarmac also lead to problems with the manual threshold tuning. To counter this i used a filtering of the found coefficients, radius of curvature and dist from center. This meant that problem areas would be lessened with previous values helping soften the mistakes. On top of this any result that was not consider suitable, any result were the lanes were not suitably parallel or to large a difference in radius, were not used and previous values used instead.
Filtering does have large downsides however with problems such as lag coming into play which could cause the lane prediction to be wrong for any quick changes.
3. The pipeline could face problems when tracking lanes on slops rather than the flat as it is in the project video. Again, as the values such as the perspective matrix are hardcoded  and tuned manually this means any variation ouside of this could cause problems
4. Bumps in the road also cause problems to the pipeline. It can be seen in the video that when the car hits a bump the draw mask bounces as well meaning that the image processed at that point was not suitable for anaylsis. Playing with the image i found that lanes were just too displaced from the center of the image that all of the transforms worked incorrectly. A fix to this would be adaptable transforms that could move with the lane lines if they shift in the field of view
5. Noisey gradient made it hard to produce smooth results even with filtering. I tried playing more with the thresholds to remove the noise but was unable to so am unsure on what to do here
