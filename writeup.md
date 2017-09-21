# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./center_2017_09_18_20_00_15_142.jpg "Center Lane Driving"
[image2]: ./center_2017_09_18_20_25_50_864.jpg "Recovery From Right Side"
[image3]: ./center_2017_09_18_20_26_17_648.jpg "Recovery From Left Side"
[image4]: ./center_2017_09_18_22_03_17_398.jpg "Curve Driving"
[image5]: ./center_2017_09_18_21_41_17_958.jpg "Opposite Direction Driving"
[image6]: ./left_2017_09_18_20_01_43_969_small.jpg "Left Image"
[image7]: ./center_2017_09_18_20_01_43_969_small.jpg "Center Image"
[image8]: ./right_2017_09_18_20_01_43_969_small.jpg "Right Image"
[image9]: ./right_2017_09_18_20_01_43_969.jpg "Original Image"
[image10]: ./right_2017_09_18_20_01_43_969_flipped.jpg "Flipped Image"
[image11]: ./history_training.png "Training History"


## Rubric Points
Here I consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/schambon77/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/schambon77/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/schambon77/CarND-Behavioral-Cloning-P3/blob/master/model.h5) containing a trained convolution neural network 
* [writeup.md](https://github.com/schambon77/CarND-Behavioral-Cloning-P3/blob/master/writeup.md) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](https://github.com/schambon77/CarND-Behavioral-Cloning-P3/blob/master/model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

For this project, I used the same architecture as the NVIDIA model discussed in the project videos.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The training data included recordings of driving in both directions on the track. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (2 laps), recovering from the left and right sides of the road (over 1 lap), and driving on the track in the opposite direction (2 laps) for generalization purposes.

The training data is recorded at similar speed as in autonomous mode, 9mph.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the project videos and reproduce the steps starting with a small amount of training data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set with a 80/20 ratio.

#### 2. Final Model Architecture

My final model consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 color image   							| 
| Lambda     	| normalization  	|
| Cropping					|	Removes 70 rows from top image, 25 from bottom					|
| Convolution 5x5     	| 24 filters, 2x2 stride 	|
| RELU					|												|
| Convolution 5x5     	| 36 filters, 2x2 stride 	|
| RELU					|												|
| Convolution 5x5     	| 48 filters, 2x2 stride 	|
| RELU					|												|
| Convolution 3x3     	| 64 filters, 1x1 stride 	|
| RELU					|												|
| Convolution 3x3     	| 64 filters, 1x1 stride 	|
| RELU					|												|
| Flatten 		|         									|
| Fully connected		|  outputs 120 samples       									|
| Fully connected		|  outputs 84 samples       									|
| Fully connected		|  outputs 1 samples       									|

The data is normalized through a Lambda layer, and cropped to remove unnecessary parts of the picture for driving purposes.
The model includes 5 convolutional layers followed each by RELU layers to introduce nonlinearity, and 3 fully connected layers.
The output is a single value (regression) representing the predicted steering angle.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from a situation where it is too close to the side of the track. These images show what a recovery looks like starting from:
* the right side of the track:

![Recovery From Right Side][image2]
* the left side of the track:

![Recovery From Left Side][image3]

I captured more training data along curves while driving smoothly in the center of the road:
![Curve Driving][image4]

I also recorded data while driving opposite direction in order to help the model generalize and counter the left turn bias:
![Opposite Direction Driving][image5]

I have used all 3 images recorded at each timestamp as shown below (left, center, right, resized to be displayed side by side):
![Left Image][image6]
![Center Image][image7]
![Right Image][image8]

To augment the data set, I have also flipped images and angles. For example, here is an image that has then been flipped:
![Original Image][image9]
![Flipped Image][image10]

I ended up with a set of 121224 data points which randomly shuffled, and split into training (80% - 96978 points) and validation (20% - 24246 points).

Given the size of the data set, a generator was used in order to load the data into memory in manageable batches.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was increased from 5 to 10 in order to lower the training and validation loss, although the history loss shown below indicates that 7 epochs would have been good enough. 

![Training History][image11]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Simulation

As shown in the [recorded video](https://github.com/schambon77/CarND-Behavioral-Cloning-P3/blob/master/run1.mp4), the car was able to navigate correctly in autonomous mode. In particular, it respected the constraints: no tyre leaving the drivable portion of the track surface, car didn't pop up onto ledges or roll over any surfaces considered unsafe.
