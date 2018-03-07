# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[01]: ./01.png
[02]: ./02.png
[03]: ./03.png
[04]: ./04.png
[05]: ./05.png
[06]: ./06.png
[07]: ./07.png
[08]: ./08.png
[09]: ./09.png
[10]: ./10.png
[11]: ./11.png
[12]: ./12.png
[13]: ./13.png
[14]: ./14.png
[15]: ./15.png
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (32, 32)
* The number of unique classes/labels in the data set is ? 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

![alt text][01]
![alt text][02]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduces the complexity and is processed faster. Also, the color information in case of traffic sign is not trivial. Thus, converting to Grayscale is a wise choice.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it helps to scale/shrink the image dataset at the same level. The normalization helps later in the optimization process. The optimizer (SGD, adam etc) will jump around less and will reach global minimum sooner if the data is normalized.

Apart from simple normalization, mean zero, pca etc can be used to have more stable dataset. However, I havent used it here.

I decided to generate additional data because it will give my network more features which it can learn. For eg. changing the orientation of the image, scale up or down, all these steps helps the network to be less biased and have more feature information.

To add more data to the the data set, I used the image augmentation apis from https://github.com/vxy10/ImageAugmentation. It can generate nice images with random rotation, change of brightness and other affine transformations.

Here is an example of an original image and an augmented image:
- Sample image 01 (100kph):
![alt text][03]
- Sample image 02 (20kph):
![alt text][04]

The difference between the original data set and the augmented data set is the following ... 

- The new images are rotated, with varied brightness and up/down scaled.
- The new images are almost 4 times the existing image dataset thus we have more data to train our network using the existing images.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

- | Layer         		|     Description	        					| 
- |:---------------------:|:---------------------------------------------:|
- | Input         		| 32x32x1 Grayscale image   					| 
- | Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
- | RELU					|												|
- | Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
- | Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64 	|
- | RELU					|												|       | Max pooling	      	| 2x2 stride,  outputs 8x8x64 				    |	
- | Convolution 3x3	    | 1x1 stride, same padding, outputs 8x8x32 	|       
- | RELU					|												| 
- | Max pooling	      	| 2x2 stride,  outputs 4x4x32 				    |		
- | Fully connected		| 1024       									|
- | Fully connected		| 1024       									|
- | Fully connected		| 1024       									|
- | Softmax				| 43        									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the network using different values for each hyperparameters.
The list of hyperparameters is as below:

- epochs: I started with 10 epochs, but trained my final network using 100.
- batch_size: I used 100 and 128 value for this. 128 seemed to give better result.
- learning_rate: Initially, I tried the default learning rate of Adam optimizer. However, later, I used 0.003 along with exponential decay. Ideally, Adam doesnt need this, however, I tried it for fun and it made very slight improvement so I decided to keep it.
- exponential decay: 0.9999
- beta: I used it to tune the cross entropy cross. It helps to control by how much amount you want to push the large weights.
- display step: Used to print the stats

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

- training set accuracy of ? **99.6 %**
- validation set accuracy of ? **99.5 %**
- test set accuracy of ? **93.8 %**

![alt text][15]
If an iterative approach was chosen:

1. What was the first architecture that was tried and why was it chosen?
	1. I choose it after applying things in brute force way. I added and deleted layers multiple times and observed the change in accuracy. I used read about existing high performing network structure on similar kinda image dataset and tried to use similar structure.
	2. I initially tried with 2 convolution layer and with dropout of 0.75. I realised that I am shooting lot of useful neurons and then changed the dropout value to 0.25. I also added addition convolution layer.
	3. I was using a filter size of 5x5 in all the convolution layers, however, I realised may be the size if big for the image dataset we have. Thus, switched to 3x3 filter size.
1. What were some problems with the initial architecture?
	1. The filter size was quite big thus small features like corners in the sign image were getting missed.
	2. The layers were not enough to have enough parameters to accomodate all the features. I also increased the number of filters in a layer which gave in better accuracy results.
1. How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
	1. As described above, I added/removed layers multiple times.
	2. I adjusted the learning rate and exponential decay parameters for the Optimizer.
	3. I also changed the batch size value to see how it affect the network performance. I also had out of memory issues on AWS machine couple of times and thus had to reduce the number of filters I added in the convolution layers and also had to change the batch size. Smaller batch size and less parameters were memory friendly :)
	4. I already added RELU and dropout to avoid overfitting and for regularization. I played around with the dropout probability and observed its effect on the overall accuracy.
* Which parameters were tuned? How were they adjusted and why?
	* As decribed in the answer above, I changed all the hypermeter values to see how it affects the overall accuracy of the network. Many time, it lead to even loss in accuracy.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
	* Adding an additional convolution layer helped in the overall accuracy.
	* Introducing the beta variable to control the cross entropy loss proved helpful.
	* Small batch size was helpful in avoiding out of memory errors on the AWS machine.
	* Dropout layer prevents overfitting by reducing the complexity of the network.
	* The neurons are shooted down as per the selection of the probability value in the dropout layer.

* If a well known architecture was chosen: What architecture was chosen?
	* I played around with the various network tried on mnsit dataset. The state of the art technique is to use 2 to 3 conv layers with max pooling, dropout and relu. It has been tested by many people and the accuracy in all forms in quite excellent. So, I tried the same here.
* Why did you believe it would be relevant to the traffic sign application? How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
	* The traffic signs images are good for a basic network like mine. The color information isnt trivial thus grayscale images work well. The orientation might be an issue at times, but since I have trained the network using the augmented data, this might not be an issue. The sign images arent so big and the features are not too big or too small, i.e. the features on human faces like eye, lips corners etc requires small conv filters to read this detailed information from the face, but here 3x3 filter worked quite well.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][05] ![alt text][06] ![alt text][07] 
![alt text][08] ![alt text][09]

I downloaded the german traffic image dataset from the official website:
http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Structure

Since the images were in *.ppm format I am attaching the images generated from my assignment notebook.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The result of the prediction are shown above. ( The picture displays the test image on the left and predicted one on the right.

- | Image			        	|     Prediction	        					| 
- |:---------------------		:|:---------------------------------------------:| 
- | Vehicles over 3.5 metric tons prohibited| Vehicles over 3.5 metric tons prohibited
-  									| 
- | Speed limit (30km/h) | Speed limit (30km/h)
										|
- | Keep right| Keep right
											|
- | Turn right ahead| Turn right ahead
					 				|
- | Right-of-way at the next intersection| Right-of-way at the next intersection
      							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. The prediction values / probability values were above 0.98 in all the cases, thus the showing the high confidence level of the predictions made.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

* The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

1. The confidence level of the predictions is quite good. The prediction probability for the correct class label are higher than 0.98 in all the cases. Instead of writing the values for the test images( image from web) I am attaching the images generated in the python notebook.

1. I wanted to try batch normalization and also have added some code which I found at https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412, however, I didnt have time to incorporate it.
 
1. I also think using dilated convolution layers might be useful here as the it will increase the overall filter size and thus it might help in improving overall networks performance.

![alt text][10] ![alt text][11] ![alt text][12] 
![alt text][13] ![alt text][14]
