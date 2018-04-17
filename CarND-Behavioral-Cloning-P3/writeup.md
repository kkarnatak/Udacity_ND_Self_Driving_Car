# **Behavioral Cloning** 

## Writeup Document


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./images/original_distribution.png "Original Visualization"
[image1]: ./images/applied_uniform_distribution.png "Uniform dataset Visualization"
[image2]: ./images/arch.png "Keras Layers Summary"
[image3]: ./images/nvidia.png "NVIDIA network model"
[image4]: ./images/center.jpg "Center Lane"
[image5]: ./images/right.jpg "Right Image"
[image6]: ./images/right_recovery.jpg "Right recovery Image"
[image7]: ./images/left.jpg "Left Image"
[image8]: ./images/left_recovery.jpg "Left recovery Image"
[image9]: ./images/flip_crop.png "Flipped and cropped Image"
[image10]: ./images/converted.png "Converted"
[image11]: ./images/figure.png "Training Figure"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The process_data.py file containst the data pre-processing related code.

The Visualization.ipynb notebook containts the code to visualize/analyse the recorded image dataset before and after the augmentation.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

For my model, I have used the NVIDIA architechture in my codebase. My model consists of a convolution neural network with the following architecture.

![alt text][image3]

Ref: [https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf "NVIDIA self driving car architecture")

In addition, I have used BatchNormalization after each ELU Layer. The common tested trend in CNN shows that the combination such as: Conv2D -> Activation -> BatchNormalization -> Dropout -> Dense -> SoftMax is quite helpful and leads to good results. Thus, I sandwiched BatchNormalization layer in between Activation and Conv2D layers. The final model was quite robust and had lower validation loss. The BatchNormalization adjusts and scales the activations. This is necessary/helpful to speed up the training process. The optimizer will spend less time in running around the arbitrary dataset.

I used ELU instead of RELU as in some papers its shown that the ELU networks not only fix the mean but also the variance of the neuron activation. [https://arxiv.org/pdf/1706.02515.pdf](https://arxiv.org/pdf/1706.02515.pdf "Self-Normalizing Neural Networks"). However, in my case the validation loss didnt show much difference between ELU or RELU.

I have also used a Lambda layer to normalized the input image data. Thus, the first Convolution layer receives a normalized input. I wanted to use some preprocessing steps which I used in one of my other project [https://github.com/kkarnatak/fb_facedetection](https://github.com/kkarnatak/fb_facedetection) However, converting around 19000 images were time consuming on my laptop and transferring them to the AWS machine also took hours. Thus, I havent used them for now. I wanted to use Gabor filter in this project, as I have used in my personal project mentioned earlier. Gabor filter can make good use of temporal information and thus may be lead to some interesting results. However, not sure how useful it will be in this case. Thus, I will try to use it later ( with more time )

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line #45). 

Dropout is a form of regularization. It constrains the network adaptation to the data at the training time and thus it avoids the network to become “too smart” in learning the input data.Thus, it helps to avoid overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

To do this, I used the sklearn `train_test_split` api. This splitted my image dataset for validation and training. This is part of cross validation approach. The idea behind is to hold out the part of available dataset and use it only for validation. Thus, the model wont be just predict the images it was shown during training, but would also predict new images correctly.

#### 3. Model parameter tuning

The model used an adam optimizer. The default learning rate of Adam is 0.003 which remained unchanged, however, I introduced learning decay : `decay=1e-6`. In general, decay is not used with Adam as it adapts the learning rate as per the calculated gradient, however, Adam uses the initial learning rate, or step size according to the original paper's terminology, while adaptively computing updates. Step size also gives an approximate bound for updates and its use is encouraged in the paper presented at NIPS last year
[https://papers.nips.cc/paper/7003-the-marginal-value-of-adaptive-gradient-methods-in-machine-learning.pdf](https://papers.nips.cc/paper/7003-the-marginal-value-of-adaptive-gradient-methods-in-machine-learning.pdf)

Thus, I used it anyway. I also used RmsProp just to try it out, however, didnt make much difference, or I couldnt notice any.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

The code was easy to write as I have worked on keras before, however, tricky part in this project was data collection. I had to drive several times on simulator to collect some meaningful data. I also looked in the forums and internet and used the tips for the data collection, which usually were centered around the recovery situations and also to drive in the opposite direction. This will make the model unbias w.r.t driving direction ( scene information ). However, flipping the images would also suffice the task.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train a Convolution neural network using the recorded data set from the simulator. The main idea was that the network will learn specific features from the scene on the track and will be able to learn the possible values of the steering angle for the similar scene frames. The input data for training was processed and augmented to keep non-trivial information and also do the uniform distribution of the dataset ( steering angles ), as most of the dataset contained the steering angle ranging [-0.1, 0.1]. See the figure below:

![alt text][image0]

Thus, it was important to do uniform distribution, following which I had the following distribution:
![alt text][image1]


My first step was to use a convolution neural network model similar to the network used in the NVIDIA paper mentioned above. I thought this model might be appropriate because they have tested the network with similar camera setup and recording techniques. And, the output of the network was also the steering angle values thus it seemed to be the right network to start with.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. My model was not overfitting since beginning as I used the cross validation and dropout layer.

The final step was to run the simulator to see how well the car was driving around track one. At first, I only used the Udacity data to train the network and then used the trained model on the track 1. However, the vehicle fell off the track at several points and couldnt recover. Then I recorded data using simulation. As suggested in the lecture video and also on the discussion forums and internet, it was vital to record the recovery situations such that the network learns how to react when its not in the center lane, which and how much variance should be in the steering angles in the current frame.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. The vehicle can drive without any problem several rounds on Track 1. However, its not stable on track 2.

#### 2. Final Model Architecture

The final model architecture (model.py lines 26-73) consisted of a convolution neural network with the following layers and layer sizes :

##### Keras network structure

![alt text][image2]

I mentioned in the previous sections, I have sandwiched the BatchNormalization layer between the Conv2D and Activation layer. This is in addition to the NVIDIA network.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. One in usual direction and one in the opposite. Here is an example image of center lane driving:

![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn the steering angle value in these recovery situation. If I would train the network only on the center lane driving, it wouldnt learn how to react in this kind of recovery situations, after all, the network is as good as the data we fed to it. These images show what a recovery looks like starting from right/left edge of the road and then I drove it to the center. I repeated it many times on the corner parts and around the bridge.

**Example images:**

*Took it to the extreme right:*

![alt text][image5]

*And, back to the center:*

![alt text][image6]

**I repeated the same for the left side:**

*Took it to the extreme left:*

![alt text][image7]

*And, back to the center:*

![alt text][image8]


Then I repeated this process on track two in order to get more data points.

To augment the dataset, I also flipped images and added the steering angle correction (correction was opposite to the normal direction drive, as now left side is right) thinking that this would generate additional data without actually recording it on the simulator. For example, here is an image that has then been flipped:

![alt text][image9]

The code for the above visualization is in the notebook `Visualization.ipynb`.

I also tried to use the learning from the past projects, i.e. the binary threshold images and see how the network performs using those images:

The binary image looked as below:

![alt text][image10]

Before converting it to the binary image, I converted it to HSV color space. The idea was to increase the contrast of the frame from the video and then use the S,L and B channel threshold to create a binary image. This was done in the method
 
	def pre_process_image(input_image, convert_to_color=cv2.COLOR_BGR2YUV, convert_to_binary=False):
The binary image can be seen above "converted image".

However, converting to binary image is resource expensive. It tooks ages for the GPU to finish one epoch, thus I had to switch this conversion off for now. I even tried to convert the images to binary threshold images before starting the keras network, however, due to the huge number of images, this was also a very time consuming task. Thus, inefficient approach for now. 

After the collection process, I had `10772` number of image dataset. I also used the udacity dataset and appended my recorded data onto it. This was done in the function `def process_csv(udacity_data, my_data):`


    Udacity images:  8036
    
    Recorded images 10772
    
    Total:  18808

	Number of Training Samples: 15046
	Number of Validation Samples: 3762

I then preprocessed this data by 

I finally randomly shuffled the data set using the sklearn Shuffle api. This helps in reducing the validation loss. I finally put `3762` of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 50. I couldnt generate the plots on the AWS machine as it doesnt support 
gui and thus doesnt save the figures. The figure below is generated using my laptop and thus the epochs are only 30 and steps_per_epoch were also reduced to half. Laptop was burning already :) However, this will give rough picture of the training and validation loss. As can be seen both lines runs close to each other thus no overfitting. The high peaks in the training are as the optimizer (i.e. Adam) is looking in the search space.
![alt text][image11]

One observation: I used MSE as loss function ( as suggested in the NVIDIA paper ). I used the MSE metrics in the compile function. I used 2 machines to train the network, AWS and my personal laptop, in one I could see the mse metrics information, however, in the other one I couldnt. I believe this was due to different versions of keras. 