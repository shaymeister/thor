from tensorflow.keras.datasets import mnist
from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np

# start a session
sess = tf.compat.v1.Session()

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize our data
x_train = x_train - np.mean(x_train) / x_train.std()
x_test = x_test - np.mean(x_test) / x_test.std()

# use one-hot encoding
num_class = 10
train_labels = tf.one_hot(y_train, num_class)
test_labels = tf.one_hot(y_test, num_class)

# set our parameters
batch_size = 784
samples = 500
learning_rate = 0.03
img_width = x_train[0].shape[0]
img_height = x_train[0].shape[1]
target_size = max(train_labels) + 1
num_channels = 1
epoch = 200
no_channels = 1
conv1_features = 30
filt1_features = 15
conv2_features = 15
filt2_features = 3
max_pool_size1 = 2
max_pool_size2 = 2
fully_connected_size1 = 150

# declare placeholders
x_input_shape = (batch_size, img_width, img_height, num_channels)
x_input = tf.placeholder(tf.float32, shape = x_input_shape)
y_target = tf.placeholder(tf.float32, shape=(batch_size))
eval_input_shape = (samples, img_width, img_height, num_channels)
eval_input = tf.placeholder(tf.float32, shape = eval_input_shape)
eval_target = tf.placeholder(tf.int32, shape=(samples))
