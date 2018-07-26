## Writeup : Project 4 | Udacity SelfDriving Car Engineer ND


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

[image1]: ./images/img01.PNG "Undistorted"
[image2]: ./images/img02.png "Corner01"
[image3]: ./images/img03.png "Corner02"
[image4]: ./images/img04.png "Corner03"
[image5]: ./images/img05.png "Corner04"
[image6]: ./images/img06.png "Unwarped chessboard"
[image7]: ./images/img07.png "Undistorted lane image"
[image8]: ./images/img08.png "Sobel Gradient X and Y"
[image24]: ./images/img24.png "Threshold direction"
[image25]: ./images/img25.png "Threshold magnitude"
[image9]: ./images/img09.png "Combined Filter"
[image10]: ./images/img10.png "S-Threshold"
[image11]: ./images/img11.png "Bird Eye 01"
[image12]: ./images/img12.png "Bird Eye 02"
[image13]: ./images/img13.png "Bird Eye 03"
[image14]: ./images/img14.png "Bird Eye 04"
[image15]: ./images/img15.png "Fill Area 01"
[image16]: ./images/img16.png "Fill Area 02"
[image17]: ./images/img17.png "Fill Area 03"
[image18]: ./images/img18.png "Histogram 01"
[image19]: ./images/img19.png "Histogram 02"
[image20]: ./images/img20.png "Threshold values"
[image21]: ./images/img21.png "Threshold value"
[image22]: ./images/img22.png "Threshold value"
[image23]: ./images/img23.png "Fill lane with curvature"
[video1]: ./project_video.mp4 "Video 1"
[video2]: ./challenge_video.mp4 "Video 2"
[video3]: ./hard_challenge_video.mp4 "Video 3"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.


The code for this step is contained in the 2nd code cell of the IPython notebook located in "./Advanced_Lane_Finding.ipynb". In the code from cell #2 to #3 I do the following:

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. This is in accordance with the quizes and lecture videos.

The chessboard corners on the test chessboard images looks as below:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

This step will restore the image by using the camera matrix and distortion information dumped into the disk in the previous step.
Later, it will use Perspective Transformation and Warping APIs of OpenCv to transform the image file. Both are used for image alignment.

The Perspective Transformation calculates a perspective transform from four pairs of the corresponding points.
To do this, I calculate a correct transform by using the getPerspectiveTransform(). Take 4 points from the original image, calculate their correct position in the destination, put them in two vectors in the same order, and use them to compute the perspective transform matrix.

Note: Make sure the destination image size (third parameter for the warpPerspective()) is exactly what you want. Define it as Size(myWidth, myHeight).

The result of Perspective and warp transformation on the chessboard image is as below:

![alt text][image6]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image7]

(Note: The difference is minute, compare the dashboard boundary at the bottom of both images)
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cells 8 to 17 in the notebook file). This was done as suggested in the lecture videos and the quizes. I modified the values to fit the requirements of our test images and videos. Here's an example of my output for this step. 

The advantage of using these is to have better detection of yellow and white lane, also in shadow and dark situations or images with occulusions.

I also tried the sobel gradient thresholds along with color channel thresholds in different color spaces. The description is in the notebook. Below are the results of different sobel gradients:

![alt text][image8]

Below are the images after using the threshold magnitude and the threshold direction.

![alt text][image24]
![alt text][image25]

S-Channel threshold and combined image:

![alt text][image10]
![alt text][image9]

Combined binary image after the test image is warped and with bird-eye view:

![alt text][image20]
![alt text][image21]
![alt text][image22]

I have used the combination all channel binary images (S, L, B) and X & Y gradient in my video pipeline.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in 15 th cell code the notebook file.  The `perspective_transform()` function takes as inputs an image (`img`), and option to display the images, if required.  I chose the hardcode the source and destination points in the following manner:


	# Define the region
   	_x, _y = 360, 258
	offset_1 = 48
	offset_2 = 2
	    
	src = np.float32([[int(_x-offset_1),_y],
	                  [int(_x+offset_1),_y],
	                  [int(0+offset_2),390],
	                  [int(720-offset_2),390]])
	dst = np.float32([[0,0],[720,0],[0,405],[720,405]])


This resulted in the following source and destination points:
dst:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 312, 258      | 0, 0          | 
| 408, 258      | 720, 0        |
| 2, 390        | 0, 405        |
| 718, 390      | 720, 405      |


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After achieving a combined binary image I used it to isolate the individual lane line pixels. Later, I fit a 2nd order polynomial to each of the isolated lane lines. The histogram revealed two peaks: left and right. The good points were chosen and then were fit into the 2nd order polynomial. Below are the peaks from 2 of the many iterations:

![alt text][image18]
![alt text][image19]

The area between the lane lines was filled to mark it as driving space in the lane.


![alt text][image15]
![alt text][image16]
![alt text][image17]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the 22nd cell of my notebook file `Advanced_Lane_Finding.ipynb`

The following elements were computed using the standard formulas:

- The curvature of the left and right lane was calculated by commonly used formula 

		left_curverad = 
			((1 + (2*left_fit_cr[0]*np.max(lefty) + left_fit_cr[1])**2)**1.5)
	
		right_curverad = 
			((1 + (2*right_fit_cr[0]*np.max(righty) + right_fit_cr[1])**2)**1.5)
		
		curvature = int((Left.radius+Right.radius)/2)
	

- Postion of the car : X- intercepts of the lines.
	`distance_from_center = abs(405 - ((rightx_int+leftx_int)/2))`
	 
Since, I rescaled the image to (720, 405) in the initial step of the pipeline, I added the following condition to detect if the lane is on the left or on the right side:


	if position > 405:
		print("The car is on the left side")
	else:
		print("Car is on the right side")


where position:
`position = (rightx_cap+leftx_cap)/2`
rightx_cap and leftx_cap are the right and left intercepts.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `Advanced_Lane_Finding.ipynb` in the function `video_pipeline()` in the cell #83.  Here is an example of my result on a test image:

![alt text][image23]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The pipeline for the video has been constructed using the similar steps shared above. However, to some of the methods have been wraped up in `class PolyUtils` in cell #71. The class contains the following methods:

- **heuristic_search()**
	- This function is used when the lane was not detected successfully in the previous frames. It uses the standard sliding window technique ( shared in many links and tutorials i.e. ). The sliding window is used to detect the peaks in the histogram acquired from the combined binary threshold image as discussed in the point #2.
- **search_lane()**
	- This function is used when the lane was detected in the previous frames. Both of the above methods to slide the window in the range of 35 pixels in the direction of X-axis.
- **find_radius()**
	- This method computes the radius of the curvature as discussed in the point #5 above.
- **apply_sort()**
	- This method sorts the pixels after computing the average intercepts from the previous frames.
- **get_intercept()**
	- This method returns the intercept on the bottom and top of the axis.


The flow is as below:

1. The image is resized to 720, 405 and then undistorted.
2. The x, y gradient (sobel) and color space (b and l-channel) threshold are applied to create a binary image.
3. `search_lane` method from `PolyUtils` is called, if left or right lane pixels exists in previous frame.
4. `heuristic_search` method from `PolyUtils` is called, if no left or right lane pixels exists in previous frame.
5. Invoke numpy polyfit on the left and right pixels.
6. Sorting is applied to the detect left and right pixels.
7. Again, polynomial was fit on the existing and new detected pixels using numpy polyfit.
8. The radius of curvature for each lane (as described in point #5) was computed.
9. Get the total number of points ( left+right ) and fill the lane with green color.
10. The image was unwarped and the overlay information, i.e. the position  of the car and the radius of curvature was added on the image.

Below are the links to the output of different test videos:

1. [Link to project video result](./output_videos/project_video_result.mp4)
2. [Link to challenge video result](./output_videos/challenge_video_result.mp4)
3. [Link to hard challenge video result](./output_videos/hard_challenge_video_result.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


I used the combination of all channel binary images, i.e. S, L and B-Channel in my video pipeline. Initially, I tried using only B-channel and L-channel as the result ( refer point #2 ) from both looked sufficient to detect lanes in any condition, however, when I used S-Channel it ***increased** the lane detection performance* in the `Challenge` video. The improvement was quite visible.

However, I Was not able to use the threshold magnitude and threshold direction binary images. When I tried using them, I detected/filled lane wasnt proper. I might have to adjust few other things to add this, however, due to lack of time, I couldnt.

The polynomial fit and computing the radius of curvature was a little tricky. The lecture material helped, however, my solution does not work for the hard challenge video.

###Shortcomings

1. My pipeline works good for the video "project_video.mp4". It does okay job on the challenge video. However, for the hard challenge the circular lanes are not detected properly. To improve this, I will have to modify my pipeline to detect the arcs more accurately and improve the polynomial fit algorithms.

2. I also observed that the different opencv apis and binary image combination (S, L, B channel and gradients) is slowing down the performance. Specially, after using certain combinations of the threshold gradient and channel threshold, the cpus were highly busy. Have to optimize the processing to avoid this as during the live video processing this might induce a lot of delay ( lagging video ).

3. In the pipeline, I have not use the threshold magnitude and direction of the threshold. When I tried to use them along with other thresholds, the lane detection got fuzzy. I will try to use them correctly. The direction of the threshold might be useful in some cases.
