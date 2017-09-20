# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 13:29:22 2017

@author: Sylvain Chambon
"""

#Imports
import matplotlib.pyplot as plt
import pickle

history = pickle.load(open('training_history.p', 'rb'))
    
#plot the training and validation loss for each epoch
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.yscale('log')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('history_training.png')
