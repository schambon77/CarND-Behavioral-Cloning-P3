# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:42:49 2017

@author: Sylvain Chambon
"""

#Load data
import csv
import os
import cv2
import numpy as np

#Data loading
#============

#Helper function that recursively scans all folders from a root directory
def load_from_dir(dir):
    print('Processing directory: {}'.format(dir))
    dir_contents = os.listdir(dir)
    images = []
    measurements = []
    for dir_content in dir_contents:
        if dir_content == 'driving_log.csv':
            print('{}: found log file'.format(dir_content))
            lines = []
            with open(dir + '//' + dir_content) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    lines.append(line)
            for line in lines:
                source_path = line[0]
                filename = source_path.split('\\')[-1]
                current_path = dir + '//IMG//' + filename
                image = cv2.imread(current_path)
                images.append(image)
                measurement = float(line[3])
                measurements.append(measurement)
        elif str.startswith(dir_content, 'Training'):
            print('{}: found Training folder'.format(dir_content))
            sub_dir_images, sub_dir_measurements = load_from_dir(dir + '//' + dir_content)
            images += sub_dir_images
            measurements += sub_dir_measurements
    return images, measurements
    
#Load data; assumes data are contained in 1 or several folders called Trainingxxx
#found from the folder one level up
images, measurements = load_from_dir('..')

image_shape = images[0].shape

X_train = np.array(images).reshape((-1,)+image_shape)
y_train = np.array(measurements)

print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))

from keras.models import Sequential
from keras.layers import Flatten, Dense

#Build model
model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

#Train model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

#Save model
model.save('model.h5')