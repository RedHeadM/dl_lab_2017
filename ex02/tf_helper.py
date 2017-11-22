#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

dtype = tf.float32
USE_DTROP_OUT = True
TB_LOG_DIR = "logs"


# Tensorboard: $ tensorboard --logdir="logs" in working directory and then to praphs tab


def _conv2d(x, W):
    '''https://www.tensorflow.org/get_started/mnist/pros
        stride = 1, NO padding
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def _weight_variable(shape, stddev = 0.1 , mean = 0):
    '''https://www.tensorflow.org/get_started/mnist/pros'''
    return tf.Variable(tf.truncated_normal(shape, stddev = stddev, mean = 0))


def _bias_variable(shape):
    '''https://www.tensorflow.org/get_started/mnist/pros'''
    return tf.Variable(tf.constant(0.1, shape = shape))


def layer_conv(x, shape, actication = tf.nn.relu):
    W_conv1 = _weight_variable(shape)
    b_conv1 = _bias_variable([shape[-1]])
    return actication(_conv2d(x, W_conv1) + b_conv1)


def max_pool_2x2(x):
    '''https://www.tensorflow.org/get_started/mnist/pros'''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')


# fully conected layer with relu activation
def fully_connected_layer(x, shape, actication = tf.nn.relu):
    ''' creats a tf fully connedted layer'''
    W_fc = _weight_variable(shape)
    b = _bias_variable([shape[-1]])
    if actication is None:
        return tf.matmul(x, W_fc) + b
    else:
        return actication(tf.matmul(x, W_fc) + b)




def plt_error(error, fig_name="Training error"):
    fig = plt.figure(fig_name, figsize=(5, 5), facecolor='white')
    # plt.title(fig_name)
    fig.canvas.manager.window.raise_()  # pop windo to front
    plt.plot(error)
    plt.xlabel('#epochs')
    fig.tight_layout()
    plt.show()


def plt_valid(data_x, data_y, predivtion_y):
    fig = plt.figure("validation", figsize=(5, 5), facecolor='white')
    fig.tight_layout()
    fig.canvas.manager.window.raise_()  # pop windo to front
    plt.axes(frameon=True)
    plt.scatter(data_x, data_y, c='green', label='simulated_training_data')
    plt.scatter(data_x, predivtion_y, c='red', label='validation')
    plt.legend()
    plt.show()
