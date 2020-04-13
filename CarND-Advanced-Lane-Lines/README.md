## Project 2: Advanced Lane Finding Lines

---

[//]: # (Image References)

[image1]: ./output_images/Undist.png "Undistorted"
[image2]: ./output_images/Threshold.png "Combined Binary Threshold"
[image3]: ./output_images/X_thresh_grad.png "X Sobel"
[image4]: ./output_images/Y_thresh_grad.png "Y Sobel"
[image5]: ./output_images/Mag_img.png "Magnitude Threshold"
[image6]: ./output_images/dir_img.png "Direction Threshold"
[image7]: ./output_images/H_Color_img.png "H Color Space"
[image8]: ./output_images/L_Color_img.png "L Color Space"
[image9]: ./output_images/S_Color_img.png "S Color Space"
[image10]: ./output_images/R_Color_img.png "R Color Space"
[image11]: ./output_images/Bird_img.png "Bird's eye view"
[image12]: ./output_images/Roi_img.png "ROI Image"
[image13]: ./output_images/bin_bird.png "Binary Bird's Eye view"
[image14]: ./output_images/histogram.png "Histogram"
[image15]: ./output_images/Find_lane_img.png "Find Lane"
[image16]: ./output_images/Search_around_img.png "Search Around"
[image17]: ./output_images/Overlay_img.png "Color Overlay"
[image18]: ./output_images/Final_img.png "Final Output Image"
[image19]: ./output_images/First_image.png "Header Image"

[video1]: ./project_video_output.mp4 "Video"

![alt text][image19]


## Overview 
---

The goal for this project was to identify functions to use for an image processing pipeline to accurately detect road line markings. In the next section I will explain each function I explored and the to help create my final image processing class. 

## Project Steps:
The steps I used to create my final image processing pipeline are as follows:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a region on interest
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## Step 1. Camera Calibration

#### The goal of camera calibration is to ensure that the image going through the image processing pipeline is accurate and does not contain any distortion. This is important for accurately determining a vehicles position in world coordinates based on a given image. 

In this project, we were provided an folder of chessboard images to use for camera calibration located in `/camera_cal`

Using these images, the first step was to manually define a set of object points that would be used for each seperate image in the set of chessboard images. 

Next, for each image the function went through the following steps:
1. Read the image using 'cv2.imread'. 
2. Convert the image to gray scale using `cv2.cvtColor(img, cv2.COLOR_BG@GRAY)`
3. Find the corners in the checkerboard image using `cv2.findChessboardCorners()`
4. Create a list of the image corner points to be used in my undistortion function. 
    
This function then returns two lists: object points and image points. 



## Step 2. Apply Distortion Correction

For image distortion correction I defined a function that took an image as input and the object and image points from the camera calibration.

The function includes two steps to apply distortion correction:
1. Call the `cv2.calibrateCamera` using the object and image points from the previous function. This function returns the camera matrix, distortion coefficients, rotation and translation vectors, which will then be used in the next step. 
2. The matrices and vectors obtained from above are then used in the `cv2.undistort()` function to apply distortion correction to the input image. 
    
```python
def undist_img(img, objpoints, imgpoints):
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
```
![alt text][image1]

## Step 3. Create a Threshold Binary Image

The goal for this step was to determine which threshold strategy would work best for creating a combined binary image of edges in the scene. 

I used five different strategies and functions to create individual binary images:

Using the `cv2.Sobel()` function I was able to find:
1. Direction gradient in the x and y directions
2. Magnitude gradient
3. Directional gradient
    
Extracting the different image color information:
1. Hue, Lightness, and Saturation color space of the image: 
- Using `hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)` to convert the image to the HLS color channel. 
2. Extracting the R color space:
- Since I used the `matplotlib.image.imread()` function to read in the image, the output image is the RGB color space. From here I extracted the R color space and defined a min and maximum threshold to find the lane edges. 
   

Below are the outputs from each individual thresholding technique:
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

After spending a signifcant amount of time exploring the different outputs from each edge detection strategy, I chose to use a combination of three techniques. The direction gradient in the x direction, the S color space, and the R color space. Below you can see how I combined these three strategies in a conditional statement to find the combined binary output of the lanes within an image. 

```python
def combined_threshold(img):
    #combining thresholds of sobel_x, S channel, 
    sobel_x = abs_sobel_thresh(img, orient='x')
    s_color = hls_select(img, channel='s',thresh=(120,255))
    r_color= r_colorspace(img, thresh=(220,255))

    combined_bin = np.zeros_like(sobel_x)
    combined_bin[((sobel_x==1) & (s_color==1))| (r_color==1)]=1
    
    return combined_bin
```

![alt text][image2]

## Step 4. Apply Region of Interest

Once the combined binary image is created, the next step in my image processing pipeline was to apply a region of interest to filter out any unwanted edges in the image. 

I used a similar region of interest function that was used in the first project. I defined a trapezoid like shape in the original image and created a new blank image of zeros by creating a mask. 
`mask = np.zeros_like(img)`

This allows me to fill the blank image with the trapezoid shape:
`cv2.fillPoly(mask, [vertices], ignore_mask_color)`

Now, we can compare the original combined binary image to the the trapezoid image and only allow values in the image that are non-zero at the same indicies, which will provide only pixels within the trapezoid. This comparison using:
`masked_image = cv2.bitwise_and(img, mask)`

![alt text][image12]

## Step 5. Apply Perspective Transform 

The next step in the pipeline is to apply a perspective transform (bird's eye view) . 

This transform is completed by manually selecting the coordinates cooresponding to the shape of the lane in the original image. These coordinates are known as the source points.

Next, step is to define destination points where the source points will be transformed too. Since we want a bird eye view of the lane, we want the destination points to be a rectangle. This will help with fitting a line to the lane markings and detecting curvature on the road. 

Once both the source and destination points are obtained, the `cv2.getPerspectiveTransform` function is used. This functions returns the transformed image, and the perspective and inverse perspective matricies. 

![alt text][image11]
![alt text][image13]

## Step 6. Find the Lane Pixels

In order to find the initial pixels in a given image, the following steps were used:

1. Use a histogram to determine where the highest concentration of pixels are and begin to search the image at this point.

![alt text][image14]

2. Define the shape of a search box based on the image dimensions. Two search seperate sets of search boxes are used to find the left and right lanes respectively. The left search box is placed at the maximum value of the histogram on the left side of the midpoint. The right search box is placed in the same method on the right side of the midpoint. 

3. A sliding window method is used to find the cooresponding lane pixels. The function searches for pixels within each 'sliding window'. If the number of pixels within the window meet the minimum threshold, then indicies of each pixel is added to an array. This array is then averaged and used for the horizontal position of the next sliding window. This process is done until the entire vertical axis of the image is covered. 

4. Now that the lane pixels have been identified. A second order polynomial, `np.polyfit`, is used to calculate the line of best fit between all of the pixels found via the sliding window method. 

This function is not very cost efficient and could cause problems in line detection if used for every frame of a video. To maximize effciency, this function is only used at the very beginning of a video, or if the lines have been lost completely. 
    
In order to only call this function when needed, it is called within the search_around_poly function, which I will describe in the next section. 
   
![alt text][image15]

## Step 6.1 Detect Similar Lane Pixels

Since lanes do not jump around randomly, the assumption can be made that the lines will remain in the relative location where they have already been detected. For this reason, a function was defined to search around the already existing polynomial to find the next line marking in the frame. 

This function defines a margin area to search around the exisiting polynomial and will redraw a new polynomial once an new line marking is found. 

![alt text][image16]

## Step 7. Determine Lane Curvature and Vehicle Position

Now that the left and right line markings and polynomials have been calculated, this information can be used to determine the overall lane curvature and vehicle position relative to the lane center. 

To transform coordinates from camera to world values, I used the conversions used in the Advanced Computer Vision lesson and assumed the following (x,y) pixel to meter factors:

x pixel -> meter = 3.7/700
y pixel -> meter = 30/720

#### Determine Lane Curvature:

I used the use the equation from the second order polynomial of the left and right lines to calculate the radius of each poly nomial:

Second Order Polynomial = f(y) = Ay^2 + By + C 

Radius of curvature at any point x of function x = f(x) = Rcurve = ([1 + (dx/dy)^2]^3/2)/|d^2x/dy^2|

first derivative f'(y) = 2Ay + B
second derivative f''(y) = 2A

Radius of curvature = ([1 + (2Ay + B)^2]^3/2)/|2A^2|

To apply the conversion, the x and y factor are applied to the left and right line indicies then used to recalculate the polynomial coefficients using `np.polyfit`. The coefficients are then used to calculate the radius as shown above. 

#### Determine Vehicle Position: 

To calculate the vehicle's position to the center of the lane the following assumptions were made:

1. The camera is mounted at the vehicles center position with no offset. Therefore, the vehicle's current position is the midpoint of the image.
2. The lane center is the average point between the first indicies of the left and right lane. 

To find the distance to lane center the follow calculation was used:

```python
    f_leftx, f_lefty = leftx_pts[0]
    f_rightx, f_righty = rightx_pts[0]
    
    #vehicle center pos
    mid_img = img.shape[1]//2
    #lane center pos
    lane_pos = (f_leftx+f_rightx)/2
    
    
    distToCent = round(((mid_img - lane_pos)*xm_per_pix),2)
```


## Step 8. Warp Detected Line Back to Orginal 

To warp the bird's eye view image back to the orginal perspective image. The following steps were used:

1. Determine all the points and their indicies used for both the right and left lines.
2. Create a green polygon to show the detected lane area in the bird's eye image:
        `cv2.fillPoly(color_overlay, np.int_([all_pts]), (0,255,0))`
3. Create red lines to show the left and right lanes in the bird's eye image:
        `cv2.polylines(color_overlay, left_pts, isClosed=False, color=(255,0,0), thickness=30)`
        `cv2.polylines(color_overlay, right_pts, isClosed=False, color=(255,0,0), thickness=30)`    

4. Warp the bird's eye image back to the original perspecitve using the bird_eye_transfrom explained above. For this transform, the inverse perspective matrix is used. 

5. The add weighted function `cv2.addWeighted` is used to add the colored polygon/lines to the original image. 

![alt text][image17]

## Step 9. Output Visual Display and Lane/Vehicle Information

The final step is to add the text information about the lane and vehicle position to the final output image. 

The `cv2.putText` function is used to place text in a specific position on the final image. 
I chose to output the overall lane curvature, which is the average curvature between the left and right lines, and the vehicle position to lane center. 

![alt text][image18]

Here's a [link to my video result](./project_video.mp4)

# Discussion 

Overall, I enjoyed the challenge of this project. I learned a lot about computer vision and image processing techniques along the way.

At first, I struggled a bit to find the correct source points to use for the perspective transform. Since this was a crucial part of the image processing pipeline, I spent extra time trying to make sure I had good points to use for a correct perpective transform. 

The most challenging part of this project was trying to find a robust binary image that would be successful in every lighting and environemntal situation. With my original binary threshold image, I struggled with the difference in road color and shadows from trees or other objects in the environment. Eventually through trial and error I came up with a method that worked best for most conditions. I spent significant time comparing the binary outputs from every technique, and combining the thresholds using conditional statements. Another somewehat painful process of finding the best combined binary image, was determing the best minimum and maximun thresholds to use for each technique. I believe these thresholds could be optimized even further, but for the purpose of this project they were sufficient. 

#### Opportunities for Improvement

My pipeline for image processing certainly has areas from improvement. 

First, is the smoothing of the line calculation. The line detection can jump around slightly, causing an incorrect curvature measurement to be output. Implemented a curvature smoothing function, such as a moving average or lowpass filter, would significantly help the output of the lanes and curvature measurement. 

Second, further optimizing my binary threshold used to find accurate lane markers. Since this was already a fairly time consuming task, further improvement of this step in the image processing pipeline would improve the accuracy of lane detection. 

