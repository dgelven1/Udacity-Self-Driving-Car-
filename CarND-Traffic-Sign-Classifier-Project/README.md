# **Traffic Sign Classifier Project** 

---

For this project within the Udacity Self Driving Car course, I was tasked with creating a convolutional neural network to correctly predict the sign type of 5 randomly selected German traffic signs. I was provided three sets of data for creating the classifier. A training, validation, and a test data set. Below you will find the steps I followed to create my classifier and how my classifier performed when classifiying the random German traffic signs.

The goals / steps of this project are the following:
* Load the data set, explore, summarize, and visualize the data set (see below for links to the project data set)
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
[image10]: ./Writeup_images/CNN_Architecture.JPG
[image11]: ./Writeup_images/Dropout.JPG

Link to my traffic sign classifier [project code](https://github.com/dgelven1/Udacity-Self-Driving-Car-/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier_V4.ipynb)

## Step 1: Data Set Summary & Exploration

### Overall summary of data provided for the classifier. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### Visualize the distibution of the data set. 

Below is the distribution of the traffic sign images for the training, validation, and test data set. 

![alt text][image1]

Below is an example of some of the images the data sets contain.

![alt text][image2]

### Increase the size of training data set

After observing a large difference in the number of samples for a given sign in the distribution graph above, I wanted to give my training set a more even distribution. 

First, I chose a minimum number of images that I wanted each sign type to have. If the sign type had less than this minimum threshold, I would copy a image of the sign and add it to the distribution. I iterated through the existing number of images and made copies of each image until I reached the minimum threshold. Using this approach, I ensured that I did not make multiple copies of the same image to reach the minimum threshold, allowing the training data set to have a variety of different images for the same traffic sign type.  

Below is the distibution of training data after increasing the size of the data set:

![alt text][image3]

### Step 2: Data Augmentation and Preprocessing 
As a first step, I wanted to augment the images in the training set to give the model multiple different views of an image. The first two steps I used in the data augmentation process was to increase the sharpness and brightness of each image. After viewing a sample of the test set, I noticed the images were blurry and sometimes very dark. I decided it would be best to sharpen and brighten each image to improve the quality of the image being trained by the classifier. This step was also done to the validation and test data sets. See below for an example of how increasing the sharpness and brightness of an image can improve the clarity and quality of the image:

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


Below is an example of the different possible data augmentation techniques applied to the same image:

![alt text][image4]

### Grayscaling
After data augmentation, the next step was to convert every image to grayscale. I chose to use grayscale as a preprocessing techinque because so many of the German traffic signs have similar color patterns, which would not be benificial to use when training my model.
Furthermore, the results from multiple experiments involving traffic sign classifiers [1] have noted that using grayscale images produces a higher accuracy of classification. 

### Standardizing the Data Set

Next, I standardized the data by using the subtracting each pixel value in every image by 128, then dividing by 128. This will give the entire data set a minimum value of -1 and maximum value of 1. The goal of this standardization was for the mean and stardard deviation of data sets to be 0 and 1 respectively. 

You can see the results of applying this standardization by observing the mean and standard deviation from before and after. 
| Before         		|     After	        					| 
|:---------------------:|:---------------------------------------------:| 
| Mean = 74.26       		| Mean =  -0.41  							| 
| Standard Deviation =  82.53 	| Standard Deviation =  0.62     	|

Normal color image data has pixel values ranging from 0 to 255. So, different features in an image could potentially have a wide range of potential pixel values and could cause some difficulties while the model is learning. By standardizing the data and modifying the pixel values to be on a much small scale, this allows the model to learn more efficiently. This is seen during the trail and error of training my model. Training the model withouth standardizing the data set, the model achieved 94.9% accuracy. Using the same hyperparameters, but with a standardized data set, the model was able to achieve 96.5% accuracy. 



### Image Preprocessing Pipeline
Below is an example of how every image within the training data set is preprocessed:

![alt text][image6]

## Step 3: Design a Model Architecture

### Model architecture 
After completing the LeNet lab exercise in the course, I decided to use a similar model architecture for my traffic sign model. I began the project by running the LeNet with no data preprocessing and now changes to the arcitecture. This gave me a model accurcay of approximately 87.2%. This was a good start given no alteration to the data sets and architeture. One modification I chose to make was to add a dropout layer before each of the fully connected layers to help prevent overfitting. 

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

## Step 4: Train and Test Convolutional Neural Network Model

### Model Training
### Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, at first I used the same hyperparameters as the LeNet model given in the lab exercise. Using a dropout rate of 85% validation accuracy was pretty good at approximately 94.5%. I decided to tune the hyperparameters slightly to achieve a better validation accuracy. Below are final set of hyper parameters I used to achieve a validation accuracy above 96%. 

* Learning Rate = 0.001
* Batch Size = 64
* Epochs = 30
* Dropout Rate = 0.70

Using a smaller batch size caused the model to train much slower, but produced better validation results. Changing the learning rate did not have a big affect of the accuracy unless it was also changed along with the number of epochs. To keep the traing time to a minimum, I chose to leave the learning rate at 0.001 and increase the epochs to account for the newly introduced dropouts at each fully connected layer.  

### Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

As described above, I used an interative approach to finding a solution to achieve a validation accuracy above 93%. Below you will find a summary of the iterative steps I used to find my final solution for training my classifer:

   1. Using the LeNet Architecture and hyperparameters. Validation accuracy = 87.2%
   2. Using grayscale images instead of color images for training and validation. Validation accuracy = 91.5%
   3. Implementing standardization. Validation accuracy = 91.7% (Not a huge change, but slightly helps the accuracy)
   4. Increase the training data set size. Validation accuracy = 92.5%
   5. Implementing data augmentation to entire training data set. Validation Accuracy = 94.2%
   6. Implementing sharpening and increasing brightness of training, validation, and test datset. Validation accuracy = 95.3%
   7. Adding droput layer with a keep probability of 50%. Validation accuracy = 88%
   8. Decreasing batch size to 64 and increasing keep probability to 7%. Validation accuracy = 95.8%
   9. Increasing epochs from 10 to 30. Validation accuracy = 96.7%

### Final Model Accuracy Results
* Validation set accuracy of 96.7%
* Test set accuracy of 94.0%

### Solution Discussion

As you can see by the results above the model performs well on both the validation and test data sets. Through my iterative process I did implement a new arciteture to view the difference in accuracy results. I implemented the ConvNet arcitecture used in Sermanet and LeCun's experiment for a traffic sign classifier[1]. The architecture is similar to the LeNet but splits the output from the first convolutional layer and adds this output with the second convolutional layers output before the final full connection layer. This architecture produced better accuracy results in Sermanet and LeCun's classifer. For me this architecture produced slightly worse accuracy performance, so I decided to stick with the LeNet architeture. Below you can find an image comparing the two architectures:

![alt text][image10]

Some issues with the original LeNet architecture was overfitting. I would achieve a high accuracy on the validation data set, but the accuracy on the test set would be much lower. To solve this issue, I introduce the dropouts in between the fully connected layers. This helped the model in preventing overfitting by only keeping a certain percentage of outputs from these layers and passing them as inputs into the next layer. 

![alt text][image11]


 
## Step 5: Test a Model on New German Traffic Sign Images

### Find 6 German Traffic Signs

I used google maps street view to find traffic signs around the German city of Munich. I chose this approach to simulate how an actual vehicle would view a traffic sign in a normal driving siutation. Since the images from google street view are captured using a vehicle with a camera attached to the roof of the vehicle, I thought this would be very representative of a real driving scenario. The following 5 images were used:

![alt text][image7]

Some of these images may be difficult for the model to classify because of the similar features between multiple signs in the data set. For example the 60 kph speed limit sign could easily be misclassified as a 80 kph sign. The features of the sign are extremely similar and differentiating between a 6 and 8 might be challenging for the classifer. Also there are multiple signs that include arrows and identifying the difference between all of those signs could be difficult for my classifier. 

### Prediction Results

Below are the results of my model's prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 Kph Sign      		| 60 Kph sign   									| 
| No Entry     			|  No Entry 										|
| Stop Sign					| Stop Sign											|
| Go Straight or Left	      		| Go Straight or Left					 				|
| Yield			| Yield      							|
| Turn Left Ahead			| Turn Left Ahead      							|

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. 


### Model Prediction Probability. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below were the probabilites of each sign prediction. This probability shows the certainty of the model's predictions. As you can see below the model was 100% certain on 5 out of the 6 signs. The only sign that had less than 100% probability was the go ahead and turn left sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| 60 Kph Sign   									| 
| 1.00     				| No Entry 										|
| 1.00					| Stop Sign											|
| 0.95	      			| Go Straight or Left					 				|
| 1.00				    | Yield      							|
| 1.00				    | Turn Left Ahead      							|

The image below shows the original sign along with the top 4 possible sign predictions by the model. The sign that gave the model a challenge was the go ahead a turn left sign. This sign has similar features as many other signs, such as the go straight and turn right, ahead only or the turn left ahead sign. Even with the similarities between the features in these signs, the model correctly classified this sign and with a high probability of 95%. 

![alt text][image8]


## Discussion

Overall my convolution neural network traffic sign classifer has performed very well. With 100% prediction on images taken from google maps and real world scenario's, I am confident in my model's ability to classify signs correctly. With that said, I still think they is some room for improvement. First, different data augmentation techniques could be used to manipulate the images to train the model. Second, I could explore different network architectures with deeper models and smoother reduction of image size.


# References

[1] P. Semanet, Y. LeCun, “Traffic Sign Recognition with Multi-Scale Convolutional Networks”

[2] [Stanford CS231n](https://cs231n.github.io/neural-networks-2/)
