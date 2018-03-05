# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./image_output/02.png "solidWhiteCurve"
[image3]: ./image_output/03.png "solidWhiteRight"
[image4]: ./image_output/04.png "solidYellowCurve"
[image5]: ./image_output/05.png "solidYellowLeft"
[image6]: ./image_output/06.png "whiteCarLaneSwitch"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of the following steps:

1. I converted the images to grayscale. Grayscale conversion lowers down the complexity of the image, as we just need to focus on the edges and corner information.

2. I applied Gaussian smoothing. This resulted in a blurred image. The idea behind is to degenerate noise and keep the pixels as uniform as possible.

3. I used Canny Edge Detector to detect the edges in the image/frame by using the lower and upper threshold values. Instead of using fixed values for these thresholds, I used dynamic values by computing the median of the single channel pixel intensities of the given image. I wrote a new method canny_dyanmic_threshold_values which calculates the concerned threshold values.

4. I have also created an **auto canny** function, which calculates the values of low and high threshold on the fly.

5. I also tried to experiment with a different edge detection method **"SUSAN"**. This algorithm works quite well in certain situation and has very smooth edge detection. However, there is no python api available for this. Thus, I have to compile the c code and invoke the system call to the compiled binary. In the end, the results were not as good as canny detector. I will play around with this in future.

6. Next, I defined a Region of Interest (ROI). The idea behind ROI is to select the pixels which lies in this ROI. Here, I defined my ROI as a trapezoid. It was placed at the position where the lanes are visible or expected.

7. After ROI, I used Hough transformation to detect lines within this ROI. This gave me the basic layout of the lanes in the given scene.

8. In this step, I modified the given draw_lines() method. The intention behind this method is to filter, average and extrapolate the borders of the left and the right lanes. I performed the following steps:
	1. I calculated the slope of the lines.
	2. If the slope is negative, the detected lane is left lane, if slope is positive, the lane is right lane.
	3. I calculate the slope and intercept b of the lines.
	4. I compute the mean of the slope and the intercept.
	5. I considered lines which had a slope between 30 to 45 degrees only.
	6. I compute the border lines using the above averaged slope and intercept and plot them over the original image/frame.

9. I merge the detected lanes from the above steps over the top of the original image/frame. I declated a global variable in the notebook. This variable is "challenge" specific. For the challenge the top and bottom of the image/frame was different and also the vertices for masking were different. Thus, this was required.


### 2. Potential shortcomings with current pipeline


1. The threshold values for Canny Edge Detector were static as per the given code. The static values may be not fit for different images or video frame.
2. The current implementation uses Canny Edge Detector, however, in some cases it has been seen that Canny leaves some some edges.
3. The region of interest (ROI) is also static in the current implementation. This works for a fixed camera position, but if the camera moves the ROI will change.
4. Since we have discarded the color information completely, there are chances that the yellow lines wont be correctly detected in case of shadows, low lightening, or bad weather. This will reduce the lane detection accuracy.

#### The output from the pipelines for different test input images looks as below:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]


### 3. Possible improvements to the pipeline

1. I modified the given code to **adapt the lower and upper threshold values** for the Canny Edge Detector.
2. I propose using **SUSAN** edge detector. It has shown better (smooth edges) results over Canny in many cases. My experiment with it wasnt successful, but I havent tried hard enough yet. I expect better results from this algo.
3. To make use of the color information, we can use two color-filters:
	1. For white lane as used in this code.
	2. A new one for yellow lane.
4. The ROI should be calculated on the dynamically and should be robust. It should have  minimal influence from the lane occlusion and absence. A deep neural network like suggested in <cite>Vision-based fusion of robust lane tracking and forward vehicle detection in a real driving environment</cite> [1] could be used to make the ROI and lane detection robust.
5. We can also do some data recording and use the data to train a deep neural net for the parameters like slope, intercept to calculate the border lines on the lanes as precise as possible.


###Citations###

[1] Choi HC, Park JM, Choi WS, Oh SY. Vision-based fusion of robust lane tracking and forward vehicle detection in a real driving environment. International Journal of Automotive Technology. 2012 Jun 1;13(4):653-69.
