# **Traffic Sign Classifier Project** 

---

For this project within the Udacity Self Driving Car course, I was tasked with creating a convolutional neural network to correctly predict the sign type of 5 randomly selected German traffic signs. I was provided three sets of data for creating the classifier. A training data set, a validation data set, and a test data set. Below you will find the steps I followed to create my classifier and how my classifier performed on when classifiying the random German traffic signs.

The goals / steps of this project are the following:
* Load the data set, explore, summarize and visualize the data set (see below for links to the project data set)
* Augment and preprocess the data
* Design, train and test a convolutional neural network architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

[//]: # (Image References)

[image1]: ./Writeup_images/Data_set_histogram.png
[image2]: ./Writeup_images/Raw_Training_Data.png
[image3]: ./Writeup_images/New_Data_set_histogram.png
[image4]: ./Writeup_images/Augmented_Data.png
[image5]: ./Writeup_images/Random_Aug_Data.png
[image6]: ./Writeup_images/Image_Pipeline.png
[image7]: ./Writeup_images/My_German_Images.png
[image8]: ./Writeup_images/Top_SoftMax_Prob.png
[image9]: ./Writeup_images/Bright_Sharp_Img.png

Link to my traffic sign classifier [project code](https://github.com/dgelven1/Udacity-Self-Driving-Car-/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier_V4.ipynb)

### Step 1: Data Set Summary & Exploration

#### Overall summary of data provided for the classifier. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Visualize the distibution of the data set. 

Here is an histogram of the distribution of the training, validation, and test data set. 

![alt text][image1]

Here is an example of some of the images the data sets contain.

![alt text][image2]

#### Increase the training set to even the sample distribution
Observing a large difference in the number of samples for a given sign, I wanted to give my training set a more even distribution. 

First, I chose a minimum number of images that each sign label could have. If the sign label had less than this minimum threshold, I would copy the image of the sign and add it to the distribution. If the sign label was already greater than this threshold, I would do nothing. 

Here is an example of the distibution of training data after increasing the size of the data set:

![alt text][image3]

### Step 2: Data Augmentation and Preprocessing 

As a first step, I wanted to augment the images in the training set to give the model a wide varitey of images to train with. The first two steps I used the data augmentation was to increase the sharpness and brightness of each image. After viewing a sample of the test set, I noticed the images were blurry and sometimes very dark. I decided it would be best to sharpen and brighten each image to improve the quality of the image being trained by the classifier. This step was also done to the validation and test data sets. See below for an example of how increasing the sharpness and brightness of an image can improve the clarity and quality of the image:

![alt text][image9]


To have a robust traffic sign classifier that would be effective in real world situations, the model needs training images that resemble how traffic signs might be viewed from a vehilce. For this I implemented random data augmentation to the entire training data set. 

I decided to use five different data augmentation techniques:
| Augmentation Type    		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Random Rotation  		| Randomly rotates the image between -5 and 5 degress	| 
| Random Translation   	| Randomly translate the image left or right by  up to 5 pixels 	|
| Gaussian Blur 		| Blurs the image using a kernel size of 3 on each pixel	|
| Left Warp         	| Warps the image to left to simulate viewing sign from a left angle	|
| Right Warp	    | Warps the image to the right to simulate viewing sign from a right angle	|

Here is an example of the different possible data augmentation techniques applied to the same image:

![alt text][image4]

#### Grayscaling
After data augmentation, the next step was to convert every image to grayscale.

#### Standardizing 

Next, I standardized the data by using the subtracting each pixel value in every image by 128, then dividing by 128. This gives every pixel value a number between -1 and 1. 

You can see the results of applying this standardization by observing the mean and standard deviation from before and after. 
| Before         		|     After	        					| 
|:---------------------:|:---------------------------------------------:| 
| Mean = 74.26       		| Mean =  -0.41  							| 
| Standard Deviation =  82.53 	| Standard Deviation =  0.62     	|

#### Image Preprocessing Pipeline
Below is an example of how every image within the training data set is preprocessed before the entire set is used to train the model:

![alt text][image6]

### Step 3: Design a Model Architecture

#### Model architecture 
After completing the LeNet lab exercise in the course, I decided to use a similar model architecture for my traffic sign model. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 10x10x16   						|
| Relu		            | 									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
|Flatten output         | Flattens 5x5x16 input to a 400 output					|
|Dropout 		        |Keep rate = 85%								| 
|Fully Connected Layer  |Input 400; Output 120 				|
|RELU			        |												|
|Dropout 		        |Keep rate = 85%								| 
|Fully Connected Layer  |Input 120; Output 84 				|
|RELU			        |												|
|Dropout 		        |Keep rate = 85%								|
|Fully Connected Layer  |Input 84; Output 43 				|
|Softmax			    | 				|

![alt text][image8]

### Step 4: Train and Test Convolutional Neural Network Model

#### Model Training
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

#### Testing Model Accuracy

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Step 5: Test a Model on New German Traffic Sign Images

#### Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I used google maps street view to find traffic signs around the German city of Munich. I chose this approach to simulate how an actual vehicle would view a traffic sign in a normal driving siutation. Since the images from google street view are captured using a vehicle with a camera attached to the roof of the vehicle, I thought this would be very representative of a real driving scenario. The following 5 images were used:

![alt text][image7]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

![alt text][image8]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


