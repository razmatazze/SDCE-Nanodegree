
# Advanced Lane Line Detection

## Udacity Self-Driving-Car-Engineer Nanodegree 2017, April


### by Christoph Reiners

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


[image1]: ./images/distort_checker.png "fig1: distorted"
[image2]: ./images/undistort_checker.png "fig2: undistorted"
[image3]: ./images/road_undistorted.png "fig3: calibrated camera"


---
## Preprocessing the Image

As usual in computervision, the image we want to analyze need to be preprocessed so that algorithms can compute good results.

### Camera Calibration and Undistort the Image

First of all, an image taken by a camera is not representing the real world, but an approximation. As there is a cameralens or different camera-angles, the photographed image is distorted. For computervision algorithms this is an issue, as they solely rely on the image data. If the algorithms are used with a different camera-system they may not compute satisfying results. Therefore the camera need to be calibrated. This is done by taking pictures of an object with known parameters. Mostly a chessboard is used. First pictures of this board are taken and then transformed to what it actually should look like. For this task the opencv libary has a built-in functions.

    cv2.findChessboardCorners
    cv2.calibrateCamera
    cv2.undistort
    
The first two functions compute variables needed for the third function to output an image that has lens and perspective effects filtered out. Fig1 shows images of a chessboard and Fig2 shows the corresponding undistorted images whereas Fig3 shows an original example image of the project-video and its' undistorted version.
![][image1]
![][image2]
![][image3]



### Perspective Transformation

Another preprocessing for the task of detectinng lane-lines, is to transform the image in that way, that the road is seen from birds-eye view. With this perspective it is easy to design algorithms that give information about the slope and positions of the lanes and therefore validation of the prediction can be done. An image can be transformed by defining points in an source image, that are gonna be transformed to setpoints:


```python
def lane_transform(image):
    points_source= np.array([[leftbot],[lefttop],[righttop],[rightbot]],np.float32)
    points_desti = np.array([[leftbot_new],  [lefttop_new], [righttop_new],[rightbot_new]],np.float32)
    trans_matrix = cv2.getPerspectiveTransform(points_source, points_desti)
    image = cv2.warpPerspective(image, trans_matrix, (image.shape[1],image.shape[0]), flags=cv2.INTER_LINEAR)
    return image
```

[image4]: ./images/trapezoid_transform.png "fig4: trapezoid source points"
[image5]: ./images/transformed_lanelines.png "fig5: birds-eye-view"
[image6]: ./images/evaluation1.png "fig6:"
[image7]: ./images/evaluation2.png "fig7:"
[image8]: ./images/evaluation3.png "fig8:"
[image9]: ./images/combined.png "fig9:"

Both destination and source points are derived by taking an image and find the points that define the perspective. Fig4 shows the source points and Fig5 shows the transformed image that is scaled down a bit in this process in order to still show the lines when there is a curvature.
![][image4]
![][image5]

### Extract Lane Lines to a Binary Image

As in the first project the imagedepth is reduced to detect edges. This time not with Canny Edge Detection, but Sobel based Gradients, that are also used within the Canny Algorithm.
    
    Sobel Threshold
    Direction of the Gradient
    Magnitude of the Gradient
    
Furthermore, different approaches are tested and combined to obtain a binary black and white image representation of the camera-images. It was tested to extract the lanes from the images with a simple Colour-Threshold and a combination of the Sobel based extractions. Each was tested and parametrized on RGB-channels, R-channel of RGB-colourspace and the S-channel of the HLS-colourspace. The results of the best parametrizations on the most difficult test image are shown in Fig6 to Fig8. 
![][image6]
![][image7]
![][image8]

The best combination were the the threshold and the sobel-based-gradient combination of the R-channel that is shown in Fig9. ![][image9]


```python
def binary_lanes(image):
    # Sobel based Gradients computed on R-channel of BGR-Colourspace
    combined_img_R =sobel_combination(image,2,15,(30,90),(20,255),(90,200),(0.03,1.0))
    # Threshold of R-Channel
    threshold_img_R=col_thresh(image,2,(230,255))
    # Merge both binary pixel-extractions
    extractions_merged = np.zeros_like(threshold_img_R)
    extractions_merged[(threshold_img_R == 1) | (combined_img_R == 1)] = 1  #(threshold_img_S == 1) |
    return extractions_merged
```

[image10]: ./images/histogramm_binary.png "fig10: binary birds-eye histogram"
[image11]: ./images/windows.png "fig11: sliding windows"


---
## Detect Lane Lines

Now that there is a binary representation with nice visible lane lines, the detection algorithm can be designed. First off all the image is transformed to birds-eye-view that also serves as kind off area of interest masking.

### Detect Lane Starting Points
The starting points can be derived by analyze the sum of white pixels above the x-axis . This is called histogram. Figure10 shows this. The maximum values indicate the start of the lane as perspective wise in the nearer field of view the lanes seem to be straight.![] [image10] 

### Sliding Windows

Starting from these points, a window of set width and height is moved up to a set margin towards the mean of white pixels within the window. If the mean or the maximum margin is reached, the next window is moved. This is called sliding windows. Fitting a polynom through the center of the windows will compute a curve through data points of interest. In this case high amount of white pixels represent most likely a laneline and the sliding windows will fit a polynom according to the lanes binary representation. Fig11 visualizes this. ![] [image11] 

A basic check if the predicted window fits the reality, is to check if the direction of moving towards the mean of white pixels, fits to the expected slope of the lane. The slope will be in the direction of where a higher sum of white pixels is. 


```python
    leftx_base = np.argmax(histogram[:midpoint])
    # To identify wether it is a right slope or left slope, compute the mean value of an area around the local maxima
    # According to the image dimensions and test images, a reasonable area would be around OFFSET=80px in each direction.
    left_mean=np.mean(histogram[(leftx_base-offset):(leftx_base+offset)])
    right_mean=np.mean(histogram[(rightx_base-offset):(rightx_base+offset)])
    # Now identify, on which side more values are above a mean-based threshold are.
    # The slope is in the direction of the side with a higher number. Left-Slope=1, Right-Slope=-1
    l_datapoints=len(np.where(histogram[(leftx_base-offset):(leftx_base)]>left_mean/3)[0])
    r_datapoints=len(np.where(histogram[(leftx_base):(leftx_base+offset)]>left_mean/3)[0])

    if l_datapoints>r_datapoints:
        leftx_slope=1
    else:
        leftx_slope=-1
```

In Fig.11 for both left and right lanes it will be a slope to the right. If the window wants to slide in the oppotsite direction, the new window will instead move with the gradient of the last two windows. 


```python
    #leftx_... is the search-window-position
    #check if new position will be plausibel regarding the lane-slope
    if leftx_slope*leftx_new<leftx_current*leftx_slope:
        leftx_new=leftx_new
    #if not, use gradient of last two windows and add it to current window in regards to the slope
    else:
        leftx_new=leftx_current+(leftx_before-leftx_current)*leftx_slope
```

[image12]: ./images/margin.png "fig12: search in a margin"

This way, small noisy areas that would draft the window off-lane, can be overcome with approximate accuaracy. By no means this is a sufficient or very robust implementation, as in more noisy binary representations the direction of the sum of white pixels will not always match the actual lane slope. But with the image data that the detection shall work on, it actually is reliable.

### Detect within a Margin

When a lane is initially detected, the known information can be used to compute a new predicition faster. This is done by detecting white pixels within a margin around the previous lane polynom

    left_lane_inds  = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                       (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 

and through these pixels is a new lane-polynom fitted, Fig12. ![] [image12]
The chance is, that new polynom will drift into noisy areas as there can be a higher white pixel density. Therefore the new laneline need to be validated.

### Check if Prediction is Possible

To validate the new laneline detectioon, the derivation of the polynom is computed for a point near the start of the lane in y-direction, which is the slope. As at the lane start the lane should be nearly straight, except the car is sverwing, the slope should be near zero. As swerving can change the values significant, the sane range has to be chosen a bit wider. A slope of 1 is equal to 45°, which is set as allowed margin that the slopes can differ from one frame to another. Furthermore both lane lines should have nearly the same slope. An allowed difference of 0.25 is assumed, which equals to >10° that the slopes are allowed to differ from one another.

It is also checked if the area between the new and old prediction is too wide. For example the slopes may be ok if instead of the lane line a crack beside the lanes is detected as lane, but the predictions are kind of parallel shifted. This will be echecked with a mean squared error between the two predictions. The threshold is set to 400, as this represents a shift from about the half of lane with about 10pixels.

If the new polynom fullfills all conditions it is used in the algorithm and further stored in a set of past parameters. If the conditions are not met, a mean value of all stored past parameters is used instead. The values are stored for about the past 1.5 seconds. If there are errors for about 1.5 seconds, the lanes will be re-detected from scrath with the sliding windows method.


```python
    #new slope in amrgin of 45°  AND leftlane and rightlane slope are nearly the same
    if (left_slope_new<left_slope+1.0) & (left_slope_new>left_slope-1.0) & (abs(left_slope_new-right_slope_new)< 0.25):
        alert_slope[0]=0
        left_slope=left_slope_new
    else:
        alert_slope[0]=1

    #Further check the overall difference between old line and new one.
    if mean_squared_error(left_fit,left_fit_new)<400 :
        alert_mse[0]=0    
    else:
        alert_mse[0]=1
    
    #set the new parameters for the new lane-polynom
    if alert_slope[0]==0 & alert_mse[0]==0:
        left_fit_past.append(left_fit_new)
        left_fit=left_fit_new
    else:
        left_fit=np.mean(left_fit_past,axis=0)
```

[image13]: ./images/annotaed_lane_line.png "fig13: annotated image"

---
## Use the Information of the Lanelines

A simple usage of the Information that can be derived from the predicted lanelines are distance related informations. Suggested were to calculate the radius of curvature and the offset from the lane-center. The calculated results are in pixels though and need to be scaled to the approximate distances of the image representation. A lane is minimum 3.7 meters wide, and in the repository a lane length of 30 meters was stated for the birds-eye perspective transformated image. In pixels the lane width is approximated by about 665pixels and the lane length is about 720pixels. Fig.13 shows an example image that is annotaed with both values and a plotted lane-area. ![] [image13]


---
## Implementation of the Pipeline

The following video shows the implementation of the pipeline on the project video:


```python
%%HTML
<video width="900" height="500" controls>
  <source src="project_video.mp4" type="video/mp4">
</video>
```


<video width="900" height="500" controls>
  <source src="project_video.mp4" type="video/mp4">
</video>


---
## Discussion

The linked video works mostly reliable. On worn out areas there is a bit of jitter but the lane lines return to a stable and correct prediction fast. But already these few seconds show where the limitations are. The detection is by far not reliable. The pavement and lanelines are mostly in good shape, the lightning conditions are really good and there is no mud or rain or puddles. Also no other vehicle turns into area of interest. These are all factors where the detection will most likely fail miserably. Too bad, the final code version only accepts the project video, all the other provided videos from the repository generate an IndexError I can't resolve so I can't test my final code on these, which actual feature these bad conditions.

Also, the smooth out strategy for badly detected lane lines is not that advanced. Before implementing the "take average of previous polynoms" the overall performance was better, but at the spots where the jitter begins, it detected a wrong left lane which was bend at the end to the left, very dominant and for about 1-2 seconds. Now it is only short jitter, but at times when a good detection is shortly lost and it takes the average of past polynoms (0:25s), it is seen that this is not the best approach as it is not match with the reality. Especially on a more curvy road this will cause a big error.

To resolve this a more complex logic could be implemented, like check if the other lane is confidently stable in the past and take this polynom and shift it and transform it in regards of the lane position perspectively on the badly detected lane.

Furthermore the slope on more than just the starting point should be checked.

To conclude: the detection derived from a binary image that is transformed to birds-eye-view is relativ easy to implement and grants a good result under good conditions. Bad conditions ask for advanced processing of the predictions to derive valuable and reliable informations. Checking for possibility of the results and also generating lane predictions from the confident detections from the past or other informations that can be derived from the image have to be implement. Solely relying on the binary lane representation will not grant good results under bad conditions.


