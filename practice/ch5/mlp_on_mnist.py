# -----------------------------------------------------------------------------
# Created by: Shay Snyder
# Created on: June 22, 2020
# Last Modified: June 22, 2020
# -----------------------------------------------------------------------------
# Filename: mlp_on_mnist.py
# Project Name: MLPs on MNIST Data
# -----------------------------------------------------------------------------

# Dependencies
import numpy as np
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

np.random.seed(100) # set seed so our results are reproducible
batch_size = 256 # number of images in each optimization step
nb_classes = 10 # one class per digit
nb_epoch = 40 # number of times the whole dataset is used to learn

# load the mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the data for MLP
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# convert the data to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
