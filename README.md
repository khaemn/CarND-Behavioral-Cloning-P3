# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains files of the completed Behavioral Cloning Project (without training data).

In this project, I use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. I have trained, validated and tested a model using Keras. The model outputs a steering angle to an autonomous vehicle.

To meet specifications, the project contains these files: 
* [model.py](model.py) (script used to create and train the model)
* [drive.py](drive.py) (script to drive the car)
* [model.h5](model.h5) (a trained Keras model) and saved [weights](saved_weights.h5) file.
* a report [writeup file](writeup.md)
* [video.mp4](video.mp4) (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

#### Writeup
---
[Writeup](writeup.md) includes the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as a description of how I addressed each point.  There is also a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  Images in the writeup demonstrate how the code works with examples.

#### The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:
- TensorFlow 2.3, h5py
- numpy, scikit-learn
- cv2
- socketio, eventlet, Flask, BytesIO

## Details About Files In This Directory

### `model.py`

When is being run as

```sh
python model.py
```

trains a model and save its weights to a `saved_weights.h5` file. To convert the weights to a Keras model, please run

```sh
python restore_weights_to_model.py
```

it will create a `model.h5` file, that can be used for driving a vehicle in autonomous mode in the simulator.

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`.
Once the model has been created and saved (see above), it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

