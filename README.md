# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains all files related to my submission for the Behavioral Cloning Project.

In this project, we use what we've learned about deep neural networks and convolutional neural networks to clone driving behavior. We need to train, validate and test a model using Keras. The model outputs a steering angle to an autonomous vehicle.

It is provided a simulator where we can steer a car around a track for data collection. Image data and steering angles are used to train a neural network and then this model is used to drive the car autonomously around the track.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

Objectives
---
To meet specifications, the project requires submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of our vehicle driving autonomously around the track for at least one full lap)

Project Files
---
### [model.py](https://github.com/schambon77/CarND-Behavioral-Cloning-P3/blob/master/model.py): source code detailing the architecture and training steps of the model
### [model.h5](https://github.com/schambon77/CarND-Behavioral-Cloning-P3/blob/master/model.h5): saved trained model
### [drive.py](https://github.com/schambon77/CarND-Behavioral-Cloning-P3/blob/master/drive.py): file used to drive the car in autonomous mode
### [writeup.md](https://github.com/schambon77/CarND-Behavioral-Cloning-P3/blob/master/writeup.md): a report detailing insights on how the project was carried out
### [run1.mp4](https://github.com/schambon77/CarND-Behavioral-Cloning-P3/blob/master/run1.mp4): a video recording of one lap on track 1 in autonomous mode
