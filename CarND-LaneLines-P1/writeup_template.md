# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I smooth the image by using a gaussian filter. After the image is filtered, I use the canny edge detection to detect the edges within the image. After the edges are detected, I define a region of interest where I want to search for possible lane markers. This region of interest is a rectangle starting at the bottom of the image and converges to a area close to the center of the image. After the ROI is established, the Hough transform to search the image for a collection of points that would be possible line markers. Once these lines were determined through the hough transform and drawn on the image using the draw_lines() function. I overlayed the lines output from the draw_lines(function) on the original image using the weighted_img() function. 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by 

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
