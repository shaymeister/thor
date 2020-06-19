# -----------------------------------------------------------------------------
# Created by: Shay Snyder, snyderse@outlook.com
# Date Created: June 4, 2020
# -----------------------------------------------------------------------------
# Filename: mlp.py
# Project: Multilayer Perceptron in Tensorflow
# -----------------------------------------------------------------------------

# dependencies
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

# load the iris dataset
iris = datasets.load_iris()

# extrapilate the predictors
predictors_vals = np.array([predictor[0:2] for predictor in iris.data])

# extrapilate targets
target_vals = np.array([1. if;q
    ])
