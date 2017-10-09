**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1_car]: ./output_images/car.png
[image1_notcar]: ./output_images/notcar.png
[image2_0]: ./output_images/hog_0.png
[image2_1]: ./output_images/hog_1.png
[image2_2]: ./output_images/hog_2.png
[image3_1]: ./output_images/1_0_sliding_windows_test1.jpg
[image3_2]: ./output_images/2_0_sliding_windows_test1.jpg
[image3_3]: ./output_images/3_0_sliding_windows_test1.jpg
[image4_1]: ./output_images/sliding_windows_result_test1.jpg
[image4_2]: ./output_images/sliding_windows_result_test2.jpg
[image4_3]: ./output_images/sliding_windows_result_test3.jpg
[image4_4]: ./output_images/sliding_windows_result_test4.jpg
[image4_5]: ./output_images/sliding_windows_result_test5.jpg
[image4_6]: ./output_images/sliding_windows_result_test6.jpg
[image5_1_f]: ./output_images/single_detect_10.png
[image5_1_h]: ./output_images/single_heat_10.png
[image5_2_f]: ./output_images/single_detect_11.png
[image5_2_h]: ./output_images/single_heat_11.png
[image5_3_f]: ./output_images/single_detect_12.png
[image5_3_h]: ./output_images/single_heat_12.png
[image5_int]: ./output_images/combined_heat_4.png
[image6]: ./output_images/label_heat_4.png
[image7]: ./output_images/combined_detect_4.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file called `train_svc.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of `vehicle` image:

![alt text][image1_car]

and `non-vehicle` image:

![alt text][image1_notcar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, using all three channels (Y, Cr, Cb channels respectively): 

![alt text][image2_0]

![alt text][image2_1]

![alt text][image2_2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. The choice of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` in `YCrCb` color space seems to give the best trade-off of accuracy and speed.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using default parameters. I decided to use HOG features of all three channels, but not to use color or spatial features, for better efficiency.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a sliding window search (Line #61 - #116). I chose the following scales: 1.0, 1.5, 2.0, 2.5, 3.0. Adjacent windows have 75% overlap in both x and y directions. 

The following image shows all windows at scale 1.0:

![alt text][image3_1]

The following image shows all windows at scale 2.0:

![alt text][image3_2]

The following image shows all windows at scale 3.0:

![alt text][image3_3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

For each scale, I set the starting y position to be 400, and ending y position to be int(ystart + scale * 64 * 1.5), which is about three rows of windows. This sets up ROI and avoids unnecessary window searches. Here are some example results:

![alt text][image4_1]
![alt text][image4_2]
![alt text][image4_3]
![alt text][image4_4]
![alt text][image4_5]
![alt text][image4_6]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in the file called `detect_vehicles.py` from Line #163 - Line #195.  

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

To make detection more robust and temporally continuous, I collected 10 consecutive frames and combined their heatmaps. This combined heatmap is then thresholded (with a higher threshold) to zero out false positives.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are a few consecutive frames and their corresponding heatmaps:

![alt text][image5_1_f]
![alt text][image5_1_h]
![alt text][image5_2_f]
![alt text][image5_2_h]
![alt text][image5_3_f]
![alt text][image5_3_h]

### Here is an integrated heatmap from 10 consecutive frames:

![alt text][image5_int]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced a few issues in this project: (1) parameter tuning: a lot of parameters need to be tuned in support vector classifier (kernel, C, gamma) and hog features (color space, hog channels, orient, pix_per_cell, cell_per_block). For SVC, I tested Grid Search approach to explore different parameter settings, but the speed becomes a bottleneck in both training and testing phases. (2) detection accuracy: a few false positives and false negatives occur in some frames of the video; it is difficult to figure when the detector fails; (3) temporal continuity: to compromise detection issues, historical information is utilized to `smooth` detection result.

Due to the use of historical information, the pipeline is likely to fail when the targets move too fast. Also, different lighting conditions might confuse the detector. 

To make it more robust, I may consider adding more features other than HOG features (gradient-based), for example, color-based features. 
