# -----------------------------------------------------------------------------
# Creator: Shay Snyder, snyderse@outlook.com
# Date Created: June 1, 2020
# Date Last Modified: June 1, 2020
# -----------------------------------------------------------------------------
# Project: Deep Learning w/ Applications Using Python
# Exercise: Linear Regression, the TensorFlow Way
# -----------------------------------------------------------------------------

# Dependencies
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

# load the iris data set from sklearn
iris = datasets.load_iris()

# disable eager execution
tf.compat.v1.disable_eager_execution()

# X is the sepal length; Y is petal length
predictors_vals = np.array([predictors[0] for predictors in iris.data])
target_vals = np.array([predictors[2] for predictors in iris.data])

# split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(
                                        predictors_vals,
                                        target_vals,
                                        test_size = 0.2,
                                        random_state = 12
)

# initialize placeholders for the predictors
predictor = tf.compat.v1.placeholder(
    shape=[None, 1],
    dtype=tf.float32
)
target = tf.compat.v1.placeholder(
    shape=[None, 1],
    dtype=tf.float32
)

# create weight and bias variables
weight = tf.Variable(tf.zeros(shape=[1,1]))
bias = tf.Variable(tf.ones(shape=[1,1]))

# declare the linear model
model_output = tf.add(tf.matmul(predictor, weight), bias)

# declare loss function
loss = tf.reduce_mean(tf.abs(target - model_output))

# declare optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

# initialize the session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# fit the model using training loops
lossArray = []
batch_size = 40
for i in range(200):
    rand_rows = np.random.randint(0, len(x_train) - 1, size = batch_size)
    batch_X = np.transpose([x_train[rand_rows]])
    batch_Y = np.transpose([y_train[rand_rows]])
    sess.run(train_step, feed_dict = {predictor : batch_X, target : batch_Y})
    batchLoss = sess.run(loss, feed_dict = {predictor : batch_X, target : batch_Y})
    lossArray.append(batchLoss)
    if (i + 1) % 50 == 0:
        print('Step Number ' + str(i + 1) + ' Weight =', str(sess.run(weight)) + ' Bias = ' + str(sess.run(bias)))
        print('L1 Loss = ' + str(batchLoss))

[slope] = sess.run(weight)
[y_intercept] = sess.run(bias)

# check and display the results on the test data
lossArray = []
batch_size = 30
for i in range(200):
    rand_rows = np.random.randint(0, len(x_test) - 1, size = batch_size)
    batch_X = np.transpose([x_test[rand_rows]])
    batch_Y = np.transpose([y_test[rand_rows]])
    sess.run(train_step, feed_dict = {predictor : batch_X, target : batch_Y})
    batchLoss = sess.run(loss, feed_dict = {predictor : batch_X, target : batch_Y})
    lossArray.append(batchLoss)
    if (i + 1) % 50 == 0:
        print('Step Number ' + str(i + 1) + ' Weight =', str(sess.run(weight)) + ' Bias = ' + str(sess.run(bias)))
        print('L1 Loss = ' + str(batchLoss))

[slope] = sess.run(weight)
[y_intercept] = sess.run(bias)
