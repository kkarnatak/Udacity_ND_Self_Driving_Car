## Writeup 


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/img01.png
[image2]: ./images/img02.png
[image3]: ./images/img03.jpg
[image4]: ./images/img04.jpg
[image5]: ./images/img05.png
[image6]: ./images/img06.png
[image7]: ./images/img07.png
[image8]: ./images/img08.png
[image9]: ./images/img09.png
[image10]: ./images/img10.png
[image11]: ./images/img11.png
[image12]: ./images/img12.png
[image13]: ./images/img13.png
[image14]: ./images/img14.png
[image15]: ./images/img15.png
[image16]: ./images/scale1.png
[image17]: ./images/scale2.png
[image18]: ./images/scale3.png

[image19]: ./images/original01.png
[image20]: ./images/heatmap01.png
[image21]: ./images/original2.png
[image22]: ./images/heatmap2.png
[image23]: ./images/original3.png
[image24]: ./images/heatmap3.png
[image25]: ./images/original4.png
[image26]: ./images/heatmap4.png
[image27]: ./images/original5.png
[image28]: ./images/heatmap5.png
[image29]: ./images/original6.png
[image30]: ./images/heatmap6.png
[image31]: ./images/lane.png

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

This project was comparatively easy as most of the steps and code was thoroughly discussed in the lectures. I have used most of the code from the lecture itself. Mostly it was tuning the parameters and fitting everything well together.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `Vehicle detection and tracking.ipynb`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

We can tune the different HOG parameters, namely the number of orientations, pixels_per_cell , and cells_per_block for a single channel of the given image.
 
* Number of orientations: the number of orientation bins that the gradients of the pixels of each cell will be split up in the histogram.
* pixels_per_cells: the number of pixels of each row and column per cell over each gradient the histogram is computed.
* cells_per_block: the local area over which the histogram counts in a given cell will be normalized. Having this parameter is said to generally lead to a more robust feature set. 
* transform_sqrt: A normalization scheme called transform_sqrt to reduce the effects of shadows and illumination variations.


In a well-normalized histograms of oriented gradients (HOG). Local gradients can be binned in accordance with the orientation, can be weighted depending on magnitude, within a spatial grids of cells with overlapping blockwise
contrast normalization. Within each overlapping block of cells, a feature vector is obtained by sampling the histograms from the contributing spatial cells. The feature vectors for all overlapping blocks are concatenated to produce the final
feature vector which is fed to the classifier. The average gradient output for the car and non car image is shown above.

I tried various combinations of parameters. The combination is listed in the  table below:

![alt text][image12]

As can be seen above, the best result (overall including accuracy, time and number of extracted features) was from the first row. Thus, the parameters I choose for HOG are:

    # Feature extraction parameters
    colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the 1st configuration mentioned in the table above. It gave an accuracy of 98.51% and was the best in terms of combination with the time required for the training. For the performance of linear SVC on the other configurations please refer to the table above.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?


I used the method from the lecture and modified the scale to see how it behaves on the test images. As the feature extraction was already included, the method works as below:

1. Extract the HOG features are extracted for the entire image.
2. Subsample the features as per the given window size.
3. Classifier uses the above features for prediction.
4. The method returns the list of objects ( marked as rectangle).

The method `single_img_features` can be found in the code cell 121 in the file `Vehicle_detection_tracking.ipynb`

The image below shows the output of using different size of window on the test image:

![alt text][image16]
![alt text][image17]
![alt text][image18]

At the end, I choose the following scales and Y start and stop pixels, with an ovrelap of 75%.

	y_start = [
	400,
	416,
	400,
	432,
	400,
	432,
	400,
	464
	]
	
	y_stop = [
	464,
	480,
	496,
	528,
	528,
	560,
	596,
	660
	]
	
	scale = [
	1.0,
	1.0,
	1.5,
	1.5,
	2.0,
	2.0,
	3.5,
	3.5
	]

The scales were selected after checking the number of false positives. This was confirmed using the generated heatmaps. The heatmaps works as below:

1. Since I already have detected objects and their position, in the heatmap, add 1 to all pixels in the bounding box of the object.

    `heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1`

2. I apply the threshold by setting all the pixels which are below a certain threshold value in the heatmap to be 0. This can be chosen upto 1. This will ensure that the region inside the bounding box will remain in the heatmap. This is done as below:
    
    `heatmap[heatmap <= threshold] = 0`

For the images I choose the threshold to be 1, however, in the video pipeline, implementation was different. This will be discussed in the later section.

The images of the generated heatmap is given below:

The heatmap generated from the test image above:

![alt text][image8]

The heatmap after applying threshold:

![alt text][image9]

Below is the labelled image using the scipy label api.  It just distinguish and laterassign a label ('number') to each of the regions in an image. So, it has 2 outputs: The number of regions and an array with the same shape as the input one with the different regions numbered.

![alt text][image10]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features and histograms of color in the feature vector, which provided a nice result.

The SVM classifier needed some optimization before getting the final output. I tried with different color channels, I even tried using GRAY image ( as suggested in some forum) however, it did not serve well in my case. Thus, I sticked to YUV color space. I used all the channels to achieve overall accuracy of 98.51%. As mentioned in the section 

Other optimization techniques included changes to window sizing and overlap as described above, and lowering the heatmap threshold to improve accuracy of the detection (higher threshold values tended to underestimate the size of the vehicle).

I also tried to combine the result of pipeline for this project with the one used in Project 4 ( Advance lane finding ). This gave the following output:


![alt text][image31]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

An extra change required for the video was to keep the previous frame. This was done by implementing a small class in the cell number: `DetectedObjects`

For the video implementation the threshold for the heatmaps was computed using:
    `1 + len(d_objs.prev_box)//2`

Here are six frames and their corresponding labels generated from heatmaps: (The heatmaps are explained and shown in the section above )


Original Image:
![alt text][image19]

Labels from Heatmap:
![alt text][image20]

Original Image:
![alt text][image21]

Labels from Heatmap(None as no cars):
![alt text][image22]

Original Image:
![alt text][image23]

Labels from Heatmap:
![alt text][image24]

Original Image:
![alt text][image25]

Labels from Heatmap:
![alt text][image26]

Original Image:
![alt text][image27]

Labels from Heatmap:
![alt text][image28]

Original Image:
![alt text][image29]

Labels from Heatmap:
![alt text][image30]


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
Above

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image11]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline 
The detection of cars using SVN and sliding windows is quite powerful approach, however, there were instances when you try to have a fast classifier, it triggers not just on the cars but also on other parts of the given image. The CNN would yield better results and from the papers from Geoff Hinton's group ( [http://www.cs.toronto.edu/~hinton/](http://www.cs.toronto.edu/~hinton/) clearly shows neural networks are clear winner over SVM. However, there are papers which combines SVM and CNN, i.e. preprocess data -> generate features -> train SVM -> Use the trained features as input to CNN.

This being said, the performance of the algorithm depends on various parameters and especially the input data.

In my case, I think the performance is quite slow, as I have merged my code from the previous project (Project 4) directly with this project. Thus, I didnt optimized the codebase and might be running multiple lines twice or more. I will improve this in future and will try to make three separate classes, i.e. "Lane Detection", "Object Detection" and "FrameProcessing". This way I can process the frame and use the same instance to do lane and object detection.

Also, I believe using the Udacity labelled data might be useful is certain scenarios as it includes more detailed information about the images.