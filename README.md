# Udacity Nanodegree Self-Driving Car Engineer
## Cohort 2, started January 2017

This repository contains some of the submitted and passed files of the coding-projects of the Udacity Nanodegree Self-Driving Car Engineer.

The files for each project are minimized here to mostly only the writeup/documentation of the project and the project's code (that was developed within a jupyternotebook, that was provided by Udacity as a To-Do-list with instructions).

# Term 1
## Computer Vision and Deep Learning

## Project 1: Finding Lane Lines on the Road
Detect Lane Lines just through image processing first for a single image and finally on a test-video.
Image processing tools used were: Changing Colourspaces, Colour Selection, Region Masking, Canny Edge Detection, Hough Transformation

Coded in **Python**, notable libary used: **OpenCV**

## Project 2: Traffic Sign Classifier
Create and train a convolutional neural network to classify traffic signs from the [German Traffic SIgn Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
Preprocess the images from the dataset so that the neural network can handle them and actual learn characteristics in them to classify them.

Coded in **Python**, notable libary used: **tensorflow**, **scikit-learn**

## Project 3: Behavioral Cloning

Create and train a deep neural network to steer a virtual car autonomously around a virtual track. Input/training data for the network are car-vision images with associated steering angles. The images can also be taken from multiple camera angles and can and should be preprocessed. Training data has to be generated through driving on the track with various strategies. The trained network then ouputs steering angles in realtime to the virtual car according to the processed car-vision-image provided by the testing-enviroment/game.

Coded in **Python**, notable libary and tools used: **keras**, **cuda**

## Project 4: Advanced Lane Finding
Improving the lane detection from Project 1 using sliding window search, and implementing advanced features like measuring curvature or position of the car within in the lane.
Camera images are also corrected from distortion effects.

Coded in **Python**

## Project 5: Vehicle Detection and Tracking
Extract features from images using different colourspaces, histograms of colour camparisons, histogram of oriented gradient. Train a classifier with the extracted features to classify cars.
Then implement a sliding window search with multi-scale windows to detect cars in an image. In the next step implement functions like a heatmap to deal with multiple detections of a single car, false positives and losing a detected but still visible car and make the detection robust over multiple frames and finally apply the classifier on a video.

Coded in **Python**, notable libary used:  **OpenCV**, **scikit-learn**, **LinearSVC**, **StandardScaler**
