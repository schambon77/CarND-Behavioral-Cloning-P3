# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:42:49 2017

@author: Sylvain Chambon
"""

#Load data
import csv
import os
import cv2

def load_from_dir(dir):
    print('Processing directory: {}'.format(dir))
    dir_contents = os.listdir(dir)
    images = []
    measurements = []
    for dir_content in dir_contents:
        if dir_content == 'driving_log.csv':
            print('{}: found log file'.format(dir_content))
            lines = []
            with open(dir + '\\' + dir_content) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    lines.append(line)
            for line in lines:
                source_path = line[0]
                filename = source_path.split('/')[-1]
                current_path = dir + '\\IMG' + filename
                image = cv2.imread(current_path)
                images.append(image)
                measurement = float(line[3])
                measurements.append(measurement)
        elif str.startswith(dir_content, 'Training'):
            print('{}: found Training folder'.format(dir_content))
            sub_dir_images, sub_dir_measurements = load_from_dir(dir + '\\' + dir_content)
            images += sub_dir_images
            measurements += sub_dir_measurements
    return images, measurements
    
images, measurements = load_from_dir('..')