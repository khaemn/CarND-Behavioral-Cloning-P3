# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model_plot]: ./images/model_plot.png "Model Visualization"
[image_lo]: ./images/left_2020_12_04_19_28_57_373.jpg "Left camera original image"
[image_co]: ./images/center_2020_12_04_19_28_57_373.jpg "Center camera original image"
[image_ro]: ./images/right_2020_12_04_19_28_57_373.jpg "Right camera original image"
[image_lf]: ./images/flipped_left_2020_12_04_19_28_57_373.jpg "Left train (flipped) image"
[image_lt]: ./images/cropped_left_2020_12_04_19_28_57_373.jpg "Left train image"
[image_cf]: ./images/flipped_center_2020_12_04_19_28_57_373.jpg "Center train (flipped) image"
[image_ct]: ./images/cropped_center_2020_12_04_19_28_57_373.jpg "Center train image"
[image_rf]: ./images/flipped_right_2020_12_04_19_28_57_373.jpg "Right train (flipped) image"
[image_rt]: ./images/cropped_right_2020_12_04_19_28_57_373.jpg "Right train image"
[image_rec_co_1]: ./images/center_2020_12_21_19_04_51_028.jpg "Recovery image"
[image_rec_co_2]: ./images/center_2020_12_21_19_06_25_209.jpg "Recovery image"
[image_rec_co_3]: ./images/center_2020_12_21_19_06_51_857.jpg "Recovery image"
[image_rec_ct_1]: ./images/cropped_center_2020_12_21_19_04_51_028.jpg "Recovery image"
[image_rec_ct_2]: ./images/cropped_center_2020_12_21_19_06_25_209.jpg "Recovery image"
[image_rec_ct_3]: ./images/cropped_center_2020_12_21_19_06_51_857.jpg "Recovery image"
[image_rec_0]: ./images/recovery_0.jpg "Recovery turn sample"
[image_rec_1]: ./images/recovery_1.jpg "Recovery turn sample"
[image_rec_2]: ./images/recovery_2.jpg "Recovery turn sample"
[image_rec_3]: ./images/recovery_3.jpg "Recovery turn sample"
[image_mud]: ./images/mud_road.jpg "Mud road"
[image_mud_gray]: ./images/mud_road_grayscaled.jpg "Mud road grayscale"
[image_mud_gray_cropped]: ./images/mud_road_grayscaled_cropped.jpg "Mud road grayscale cropped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 8 and 64, a Flatten layer and three Dense layers (model.py lines 121-146).

The model also includes maxpooling layers to introduce nonlinearity (code lines 127, 131, ...), each Conv2D layer has a RELU activation for the same reason.
The data is not being normalized inside the model itself, I use some external Python code both in the training script and the driving script instead.

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting (model.py lines 123, 138). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 173, 174). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model has used an adam optimizer, so the learning rate was not tuned manually (model.py line 146).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving in the opposite direction (clockwise on the main track) 

For details about how I have created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use LeNet architecture.

My first step was to use a convolution neural network model similar to the classic LeNet, with grayscaled input, but with more Conv2D filters in each layer and of course only a single Dense output, predicting the steering angle as a continuous value (instead of 1-0 classification outputs). I decided this kind of model might to be appropriate because it worked well for various image processing tasks (as numbers recognition, road signs recognition and so on).

###### Grayscaling

The reason for grayscaling was to shrink the model as small as possible, e.g. why to process 3 channels when everything we need could be merged to 1 channel. The road boundaries are high-contrast both in colored and grayscale image, so I was thinking it should work fine in grayscale, and of course the smaller model would be trained faster and the inference will require less computations.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model with adding a Dropout layer right after the first Conv-Maxpool chunk, and after the Flatten layer. I saw such approach in some model architectures for image processing before.

Then I performed several rounds of training, observing the car behavior on the road in the worspace simulator. Unfortunately, I did not manage to set up the whole environment on my local machine (despite the simulator worked fine and the drive script was running without any errors, there was no connection between it and the sim, so it was impossible to test the model). So I tested the model by copying it to the workspace and running. It also required some tweaks, as the workspase uses the old Keras version, whil I was using the TensorFlow 2.3 with GPU support and in-built Keras to train the model locally. Thus the `model.py` contains a workaround for imports at lines 12..23, and also it was necessary to convert the trained model `.h5` file from the "new" to "older" keras format, that's why I save not the model, but only its weights, and then (inside the workspace) I restore the model, additionally using `restore_weights_to_model.py` script. The submission contains a `model.h5` file in "older" Keras format, which is compatible with Udacity workspace and can be run in it. If there is a need to run the model in TF2.3 environment, please adjust the restoring script and use `saved_weights.h5` file as the source of the trained weights.

###### Grayscaling problem

The final step after training a model was to run the simulator to see how well the car was driving around track one. In first iterations there always was a spot where the vehicle fell off the track - usually, it was right after the stone bridge, where the asphalt road turns to the left, but there is also a mud road going straight:

![alt text][image_mud]

I did not manage to overcome this while I was using a grayscaled input to the model. I suppose it can be caused by the very similar appearance of the "main" asphalt track and this mud alternative road while in grayscale:

![alt text][image_mud_gray]
![alt text][image_mud_gray_cropped]

indeed, the road looks very similar, escpecially on the cropped frame. So, in order to improve the driving behavior in these cases, I have switched back to the 3-channel full-color inputs, added some more Conv2D filters and Dense neurons, and, finally, collected a bit more data with recovery movements.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 121-145) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][model_plot]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of left, center and right lane driving:

|                     |                     |                     |
|:-------------------:|:-------------------:|:-------------------:|
|![alt text][image_lo]|![alt text][image_co]|![alt text][image_ro]|

To train the model, however, I have decided to crop the images, selecting only the bottom part of the frame (as the top part does not contain any useful information). Also, knowing about the width of the cropped image would be much larger than its heght, I decided to shrink the cropped frame vertically - as not each horizontal pixel carries a valuable information, and thus can be dropped without significant loss of information. This way, the input shape of my model is not 320x160 pixels, but 160x90, where 160 is the horizontal 320 with each second pixel dropped, and vertical size is 90 which covers pixels starting from 60 to 150 in the original image.

An example of the cropped images, that are supposed to become training data, is below:

|                     |                     |                     |
|:-------------------:|:-------------------:|:-------------------:|
|![alt text][image_lt]|![alt text][image_ct]|![alt text][image_rt]|

To augment the data sat, I also flipped images and angles thinking that this would help to increase amount of the training data. For example, here is an image that has then been flipped:

|                     |                     |                     |
|:-------------------:|:-------------------:|:-------------------:|
|![alt text][image_lf]|![alt text][image_cf]|![alt text][image_rf]|

As the Dropout layer after the first Conv-MaxPool chunk also introduces some visual noise, that is close to the salt-pepper noise, I haven't added any other noises to the training images. 

I have then also recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from any offset of the road center. These images below show what a recovery starting position looks like (original image + cropped train samples):

|                           |                           |                           |
|:-------------------------:|:-------------------------:|:-------------------------:|
|![alt text][image_rec_co_1]|![alt text][image_rec_co_2]|![alt text][image_rec_co_3]|
|![alt text][image_rec_ct_1]|![alt text][image_rec_ct_2]|![alt text][image_rec_ct_3]|

A recovery turn sample is displayed below:

![alt text][image_rec_0]
![alt text][image_rec_1]
![alt text][image_rec_2]
![alt text][image_rec_3]

Then I repeated this process on track two in order to get more data points.

After the collection process, I have merged the existing data from the workspace with my gathered data, by copying all images into a same folder and concatenating the `.csv` files' content. Overall, the `.csv` log file contained 20653 lines, which gave

```
lines * (3 images for center, left, right) * (2 for flipped and original)

20653 * 3 * 2 == 123918 
```

e.g. around 124k of data points.

For training and validation, each image was cropped (as described a bit above) and normalized by dividing over 255 -- this way each pixel was laying inside 0..1 range.
The validation splitting was performed using `sklearn.model_selection.train_test_split` function with ratio 0.2 (e.g. randomly selected 20% of the data turned into a validation set).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
