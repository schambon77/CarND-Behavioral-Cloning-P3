# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:42:49 2017

@author: Sylvain Chambon
"""

#Imports
import csv
import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
import pickle

#Constants
correction_custom_left_camera = 0.2
correction_custom_right_camera = -0.2

#Helper function that recursively scans all folders from a root directory
#Parses driving_log.csv files to store training images location and labels in memory
def load_from_dir(directory):
    print('Processing directory: {}'.format(directory))
    dir_contents = os.listdir(directory)
    samples = []
    for dir_content in dir_contents:
        if dir_content == 'driving_log.csv':
            print('{}: found log file'.format(dir_content))
            with open(directory + '//' + dir_content) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    samples.append((directory, line))
        elif str.startswith(dir_content, 'Training'):
            print('{}: found Training folder'.format(dir_content))
            sub_dir_samples = load_from_dir(directory + '//' + dir_content)
            samples += sub_dir_samples
    return samples

#Load samples info from all driving_log.csv files found
#Assumes data are contained in 1 or several folders called Trainingxxx found from the folder one level up
samples = load_from_dir('..')
print('{} samples found'.format(len(samples)))

#Split samples between train and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('{} training samples'.format(len(train_samples)))
print('{} validation samples'.format(len(validation_samples)))

#Generator to load images in memory in manageable batches
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                directory = batch_sample[0]
                line = batch_sample[1]
                for i in range(3):    #use of left and right images
                    source_path = line[i]
                    filename = source_path.split('\\')[-1]
                    current_path = directory + '//IMG//' + filename
                    image = cv2.imread(current_path)
                    #following line needed as cv2.imread() reads in BGR mode,
                    #whereas drive.py file excepts RGB images - tip found in project resources!
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    correction = 0
                    if i == 1:     #left image
                        correction = correction_custom_left_camera
                    elif i == 2:   #right image
                        correction = correction_custom_right_camera
                    measurement = float(line[3]) + correction
                    measurements.append(measurement)
                    #add augmented data through flipping
                    images.append(cv2.flip(image, 1))
                    measurements.append(measurement*-1.0)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
    
#Build model based on NVIDIA architecture explained in project guidelines
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

#Train model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, 
                                     validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, 
                                     nb_epoch=10)

#Save model and training history
model.save('model.h5')
pickle.dump(history_object.history, open('training_history.p', 'wb'))