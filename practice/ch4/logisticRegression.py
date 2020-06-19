# -----------------------------------------------------------------------------
# Created by: Shay Snyder
# Date created: June 4, 2020
# -----------------------------------------------------------------------------
# File Name: logisticRegression.py
# Project Name: Logistic Regression in Tensorflow
# -----------------------------------------------------------------------------

# dependencies
import numpy as np
import tensorflow.compat.v1 as tf
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

tf.disable_v2_behavior()

# import the iris dataset
iris = datasets.load_iris()

# extrapilate the predictors
predictors_vals = np.array([predictor[0:2] for predictor in iris.data])

# extrapilate the target
target_vals = np.array([1. if predictor--0 else 0. for predictor in iris.target])

# split the data into 75% train and 25% test
predictors_vals_train, predictors_vals_test, target_vals_train, \
    target_vals_test = train_test_split(predictors_vals, target_vals, \
                                        train_size = 0.75, random_state = 0)

# initialize the placeholders predictors and target
x_data = tf.placeholder(shape = [None, 2], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# initialize variables for weight and bias
w = tf.Variable(tf.ones(shape=[2,1]))
b = tf.Variable(tf.ones(shape=[1,1]))

# declare model: y = xW + b
model = tf.add(tf.matmul(x_data, w), b)

# declare loss function and optimizer
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = model,
                                                              labels = y_target))
opt = tf.train.AdamOptimizer(0.02)
train_step = opt.minimize(loss)

# initialize our variables and session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# make the actual prediction
prediction = tf.round(tf.sigmoid(model))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# declare the training loop
lossArray = []
trainAccuracy = []
testAccuracy = []

for i in range(1000):
    # random instances for batch size
    batch_size = 4
    batchIndex = np.random.choice(len(predictors_vals_train), size = batch_size)
    batch_x = predictors_vals_train[batchIndex]
    batch_y = np.transpose([target_vals_train[batchIndex]])
    
    # tune weight and bias while minimizing loss function
    sess.run(train_step, feed_dict={x_data : batch_x, y_target : batch_y})
    
    # loss function per generation
    batchloss = sess.run(loss, feed_dict = {x_data : batch_x, y_target : batch_y})
    lossArray.append(batchloss)
    
    # accuracy per generation on the train data
    batchAccuracyTrain = sess.run(accuracy, feed_dict = {x_data : predictors_vals_train, \
        y_target : np.transpose([target_vals_train])})
    trainAccuracy.append(batchAccuracyTrain)

    # accuracy per generation on the test data
    batchAccuracyTest = sess.run(accuracy, feed_dict = {x_data : predictors_vals_test, \
            y_target : np.transpose([target_vals_test])})
    testAccuracy.append(batchAccuracyTest)

    # print loss every 10 generations
    if (i+1)%50==0:
        print('Loss = ' + str(batchloss) + ' and Accuracy = ' + str(batchAccuracyTrain))

# plot model performance
pyplot.plot(lossArray, 'r-')
pyplot.title("Logistic Regression: Cross Entrophy Loss per Epoch")
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entrophy Loss')
pyplot.show()
