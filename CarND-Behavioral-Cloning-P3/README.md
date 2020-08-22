# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


[//]: # (Image References)

[image1]: ./writeup_images/Original_Angles.JPG
[image2]: ./writeup_images/Increased_Data.JPG
[image3]: ./writeup_images/Data_Augmentation.JPG
[image4]: ./writeup_images/Cropped_image.JPG
[image5]: ./writeup_images/OG_image.JPG
[image6]: ./writeup_images/Nvidia_model.JPG



## Overview
---

The goal of this project was to apply the deep learning knowledge gained from the Self-Driving Car NanoDegree to clone the behavior of a vehicle, which would then allow this vehicle to be driven autonomously around a track in a simulation environement. Data was collected while manually driving the vehicle around a provided test track. This data was then used as training data for the convolutional neural network. After the model was trained, the model was used to drive the vehicle autonomosly around the same test track to mimic the behavior of the actual driver.

During data collection, also known as training mode within the simulator, user generated driving data was recorded in the forms of images from three on board cameras and vehicle control data, such as steering angle, throttle, brake, and speed data. For this project, we were only interested in the camera images and steering angle data. A convolutional neural network was built using the Keras deep learning frame work. The camera and steering angle data was feed to the networ as training data, which then output final weights to be used during autonomous driving mode. 

Below you will find the data augmentation and model architecture/techniques I used to achieve these goals and successfully complete the project. 

## Steps

To complete the project and successfully drive the vehicle around the track autonomously, I followed these steps:
1. Use the simulator to collect data of good driving behavior
2. Increase the distribution size of the data
3. Implement data augmentation
4. Build a convolutional neural network using Keras
5. Train and validate the model with a training and validation set
6. Test the model to confirm the vehicle completes the test track


## Step 1: Data Collection

I put a significant effort into collecting my own data on the test track. I went with the apporach of having three entire laps of quality lane center driving data. Two normal direction and one in the reverse direction. 

Then I used two additional laps of data collection to focus on recovery to lane center. This included starting the recording with the vehicle towards the outside of the lane, then recorded the vehicles slow return to lane center to simulate the vehicle correcting itself once it got off course. 

Using this as my baseline data, I was able use it as a starting point for increase the distribution size and data augmentation techniques. 

## Step 2: Increase and manipulation of the Data Set. 

After recording the data, I wanted to visualize the distribution of the data. From my experience with prior deep learning projects, such as the traffic sign classifier project, I knew that having a wide distribution of data was important to training a successful model. 

Using a histogram I plotted the values each steering angle from the minimum to the maximum steering angle value. I used 23 bins to visualize the distribution. 

!!!!!!!!!!!!! OG data IMAGE  !!!!!!!!!!!!!!!!!!!!!!!
![alt text][image1]

From the distribution above, you can see that the number of steering angles is skewed heavilty towards zero. Meaning that the amount of images and steering angle data for close to straight line driving is very high. This would be an issue since the test track incorporate multiples curve each with a different radius of curvature. To ensure the vehicle would be able to drive autonomously in all scenarios, both straight and within curves, the distribution of this data needed to be manipulated so the training data would be more evenly distributed between curve road driving and straight road driving. 

!!!!!!!!!!!!! New data IMAGE  !!!!!!!!!!!!!!!!!!!!!!!
![alt text][image2]

I used a multi tiered method to manipulate the distribution. My goal was to decrease the difference between straight and curvature data. I achieved this by setting a limit on the more straight steering angles, so any angle below 0.2, and increasing the amount of curve data about 0.2. 

I created an empty list, then iterated through the original data information until the requirements were met for curved data and straight data. I wanted to increase sharp curve data by a factor of 10, slight curve data by a factor of 4, and increase the straight driving data by a factor of 1.5 while also setting a limit to 1000 data points. You can see in the distrubtion above, that the curve data points increased and the straight data decreased and saturated at the limit of 1000 data points. 

## Step 3: Implement Data Augmentation

Since it is impossible to cover all the situations the vehicle might encounter while recorded training data, I implmented a few image augmentation techniques to augment the images being used to train the model. I implement a random rotation, random translation, left perspective warp, and right perspective warp. These were chosen because I thought it would simulate some driving scenarios that the vehicle might encounter that would not always be recorded during the capture of data. Below are visualizations of each augmentation technique.

![alt text][image3]


## Step 4: Model Architecture

The Keras neural network library was used to create a convolutional neural network. 

I built my CNN built of the Nvidia model. Below you will find a image of the Nvidia model architecture:

![alt text][image6]

I implemented the network architecture as shown above, but added a few additional components. 

First I added my normalization layer using the Keras Lambda layer. Then adding a cropping layer to crop the image by 70 rows of pixels from the top of the image and 25 rows of pixels from the bottom of the image. 

Second, I added a layer to crop the original image. This layer crops the top 70 rows and bottom 25 rows from the image as seen in the images below:

![alt text][image4]

![alt text][image5]

Also, to prevent overfitting I added dropout layers at a drop out rate of 25%. Each dropout layer was added after each fully connected layer, with one additional dropout added in the middle of the convolutional layers. 

I chose to use the RELU as my activation function for each convolutional layer. The model was compiled using the Adam optimizer and mean squared error as the loss function. 

## Step 5: Train and Validate the Model

In order to train and validate the model, I chose to use Keras fit_generator(). Generators work great in this situation where we are working with large amounts of data. Instead of storing all of the preprocessed images and data in memory, the generator pulls one image at a time and processes that image as needed. This method is extremely more efficient than bulk storage in memory. 

During the image and data generation function is where I implemented all of my image augmentation and data manipulation. I implemented the following data manipulation steps to help improve the training of data:

* When the steering angle was above a -0.3(Requesting a left turn) I used the right camera as the training image. Vice Versa for the 0.3 steering angle

* Adjusted steering angle based on original value. Used a multi-tiered approach to adjust the steering angle. If the absolute value of the original angle was greater than 0.3, then I adjusted added a value of 0.2 to the steer angle. If the absolute value of the original angle was great than 0.1, then I added a value of 0.05 to the original value. Any measurement below these values I kept as the original.

* For every other image processed by the generator, I flipped the image to account for driving in the opposite direction. 

I chose the following parameters to use for training the model:

* Batch Size = 256

* Training Data Set Size = 20000

* Validation Data Set Size = 3000

* Dropout Rate = 25%

I chose to use one generator function for both the training and validation set. The only difference between the two generators was a boolean flag used to determine if the set was meant for training or validation. If the generator was to be used for training data, I set the flag to false to allow the random augmentation to the image. If the validation flag was set to true, the original image captured was used for training. 


## Step 6: Test the Model

Finally, the model has been trained and achieved a decent validation loss of apporximately 0.0195. Taking the model and testing it on the test track, it performed well. The vehicle stayed within the lane for the entire lap and manuevered each curve correctly. There is a high amount of waviness from the vehicle bouncing in between the lane markers, but this could be attributed to many factors. First, since I increased the amount of curvature data included in the training set this could lead to the vehicle using higher steering inputs than needed during straight line driving. Also, since I am using the left camera when turning right and vice versa for the right camera, this could lead to some over steering in certain situations when the vehicle gets close to a lane marker. 

# Summary

Overall the model completed its task of autonomously driving the vehicle around the track without departing the lane or running into any obstacle. There is still room for improvement within my model. There are many tuning factors to consider when training my model, such as the typcial CNN parameters like batch size, training data set size, drop out rate, etc. There are also data augmentation and preprocessing tuning parameters to condiser as well, for example the steering adjustment amount, at what steering angle to use the left and right cameras, what the distribution of the data set should look like, more straight or curve data points, etc. The list is very extensive for the amount of tuning that could go into this project. I believe there is a more optimal set of parameters to improve the performance of the autonomous driving around the course. 

Similar to the previous project, the Traffic Sign Classifier project, the performance of this model seemed to rely on the quality of training data rather than the model itself. Tuning the parameters of the Nvidia model only made some minor improvements. When adjusting and modifying the training data, this is when I saw the biggest benefit and change in the driving behavior of the vehicle in autonomous mode. 

Many more hours could be spend tuning both the model and data processing parameters, but for the sake of time, I chose to stop once the model achieved its goal of driving the vehicle successfully around the track without departing the lane. 

Given more time I would like to revist the project and spend time on the following topics:
* Improve the data distribution technique. I think I could implement better technique to improve the distibution of dataset while reducing the amount of data that I cut out completely. This could help with the amount of waviness observed in autonomous mode, since the majority of data that is being cut out is straight driving data. 

* Implementing a callback funciton when training the model. I went back and forth on how many epochs to use while training and often found the model overtraining when use different sets of tuning parameters. It would be nice to implement checkpoints to save the model weights during the training of each epoch. 

I thoroughly enjoyed this project and learned valuable techniques throughout. Finally getting the vehicle to successfully drive around the test track was a very rewarding feeling. I hope to revisit and improve the performance later in the course when time allows. 