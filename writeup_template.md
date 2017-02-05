#**Traffic Sign Recognition** 

##Writeup Template

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/RomainSa/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39,209
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 4

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in cells 3 to 5 of the IPython notebook.  

Let's first visualize one random example for each class:
![png](Traffic_Sign_Classifier_files/Traffic_Sign_Classifier_11_0.png)  
We can see that lighting conditions are varying

We also note that input data is highly correlated:
![png](Traffic_Sign_Classifier_files/Traffic_Sign_Classifier_13_0.png)  
We will have to take this into accound, when splitting the training data into training and validation sets. 

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cells 6 to 10 of the IPython notebook.

##### Data augmentation

The first step was to create new images of training data using scaling (zooming in and out between 90% and 110%), rotation (between -15 degrees, up to +15 degrees) and pixel shifting (between -2 and +2 pixels).  
This technique has two advantages:
- first, it fixes the unbalanced training dataset problem
- secondly, it allows the model to learn better

##### Dataset rebalancing (using augmentated data)

We rebalanced the classes to prevent the model from being biaised towards classes that are more present in the training set.  
We chose to have 5000 examples per class. New examples were made using the previously described data augmentation technique.

##### Preprocessing

We converted images to grayscale.  
This has the advantage to reduce the dimension and allows the model to be more invariant to changing lighting conditions.

We also performed a local normalisation (image by image, we substracted the mean pixel value and divided by the standard deviation). This makes the model learn faster, by having a standard range of values for the input data.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in cells 11 to 14 of the IPython notebook.  

In order to divide the training set into a training and validation sets, we must first split it by track.   
A track is a sequence of 30 images that are highly correlated (pictures taken at a few seconds of interval).  
To avoïd having this correlation between the training and validation sets, a track must not be in both the training and validation sets.

First, we calculate the track_id for each class
Then we splits the dataset between train and validation while not splitting tracks.
To finish, we shuffle the data.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the cells 15 to 18 of the ipython notebook. 

We use an implementation of LeNet5.
It is a simple model that solved the handwritten digits classification problem, which is close to ours (low-resolution images to be classified in a small number of classes).

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in cell 20 the ipython notebook. 

The Adam optimizer was used to train our model. It is an easy-to-use yet powerful optimizer. Batch size was arbitrarily set to 128.
The model was trained for 10 epochs (after 10 epopchs, the model stops to learn and tend to overfit).

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in cell 21 of the Ipython notebook.

My final model results were:
* training set accuracy of 97.4% 
* validation set accuracy of 96.4% 
* test set accuracy of 94.7% 

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six new images for the model to classify.
All seem pretty simple to classify (good lighting conditions)

![png](Traffic_Sign_Classifier_files/Traffic_Sign_Classifier_56_0.png)


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in cells 57 and 73 of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Traffic signals      		| Pedestrians   									| 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection 										|
| No entry					| No entry											|
| End of all speed and passing limits	      		| End of all speed and passing limits					 				|
| Wild animals crossing			| Wild animals crossing      							|
| Speed limit (50km/h)			| Speed limit (80km/h)      							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in cell 77 of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.61).  
The top 5 predictions are the following:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .61         			| Pedestrians   									| 
| .21     				| General caution 										|
| .18					| Road narrows on the right											|
| < .01	      			| Traffic signals					 				|
| < .01				    | Double curve      							|

The images 2 to 5 all s

The top 5 predictions for the last images are the following:


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .307         			| Speed limit (80km/h)   									| 
| .306     				| Speed limit (50km/h) 										|
| .27					| Speed limit (60km/h)											|
| .05	      			| Speed limit (120km/h)					 				|
| .04				    | Speed limit (30km/h)      							|
