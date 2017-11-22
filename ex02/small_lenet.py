#!/usr/bin/env python3

import os
import numpy as np
import time
import tensorflow as tf
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
TB_LOG_DIR = "logs"

from tf_helper import max_pool_2x2, layer_conv, fully_connected_layer
from mnist_helper import load_mnist_data


class SmallLeNet():
    #load the mnist data once
    [X_train, y_train, X_valid, y_valid, X_test, y_test] = load_mnist_data(n_train_samples = 10000, reshape = False)
    print("Training Set Shape:   {}".format(X_train.shape))
    print("Validation Set Shape: {}".format(X_valid.shape))
    print("Test Set Shape:       {}".format(X_test.shape))
    IMG_DEPTH = 1
    FILTER_WIDTH = 3

    def __init__(self, filter_num = 16, drop_out = False):
        # tf placeholder for input and output
        self.x = tf.placeholder(tf.float32, (None, 28, 28, 1))
        self.y = tf.placeholder(tf.int32, (None))
        self.one_hot_y = tf.one_hot(self.y, 10)
        self._drop_out_keep_prob = None
        self.filter_num = filter_num

        self._logits = self._create_cnn(self.x, drop_out, filter_num)

        self.last_test_accuracy = -1
        self.epoch_validation_accuracy = []
        self.epoch_test_accuracy = []


    def _feed(self,sess, X_data, y_data, train = True):
        '''can only be used in tf sess'''
        num_examples = len(X_data)
        total_accuracy = 0
        #sess = tf.get_default_session()

        if train:
            operation = self.training_operation
        else:
            operation = self.accuracy_operation

        for offset in range(0, num_examples, self.BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+self.BATCH_SIZE], y_data[offset:offset+self.BATCH_SIZE]
            batch_feed_dict = {self.x: batch_x, self.y: batch_y}

            #drop out
            if self._drop_out_keep_prob is not None and train:
                batch_feed_dict.update({self._drop_out_keep_prob: 0.5})
            elif self._drop_out_keep_prob is not None and not train:
                batch_feed_dict.update({self._drop_out_keep_prob: 1})

            accuracy = sess.run(operation, feed_dict ={self.x: batch_x, self.y: batch_y} )
            if not train:
                total_accuracy += (accuracy * len(batch_x))
        if not train:
            return total_accuracy / num_examples
        return -1

    def _run_training(self, use_gpu):
        if not use_gpu:
            config = tf.ConfigProto(
                device_count = {'GPU': 0})
        else:
            config = tf.ConfigProto()

        with tf.Session(config = config) as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(SmallLeNet.X_train)

            print("filter size {}".format(self.filter_num,))
            print("learning_rate {}".format(self.lr))
            print("optimizer {}".format(self.opt))
            if self._drop_out_keep_prob is not None:
                print("drop_out is used")
            else:
                print("drop_out is not used")

            for i in range(self.EPOCHS):

                SmallLeNet.X_train, SmallLeNet.y_train = shuffle(SmallLeNet.X_train, SmallLeNet.y_train)

                self._feed(sess,SmallLeNet.X_train, SmallLeNet.y_train, train = True)
                valid_accuracy = self._feed(sess,SmallLeNet.X_valid, SmallLeNet.y_valid, train = False)
                self.epoch_validation_accuracy.append(valid_accuracy)

                test_accuracy = self._feed(sess,SmallLeNet.X_test, SmallLeNet.y_test, train = False)
                self.epoch_test_accuracy.append(test_accuracy)
                print("epoch {} out of {} - valid_accuracy {:.3f} - test_accuracy {:.3f}".format(i+1, self.EPOCHS, valid_accuracy, test_accuracy))

            print("Test Accuracy = {:.3f}".format(test_accuracy))
            self.last_test_accuracy = test_accuracy

            #https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
            self.variables_parameter = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    def train_mnist(self, batch_size, epochs, lr, tf_opt = tf.train.GradientDescentOptimizer, use_gpu = True):
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.lr = lr
        self.opt = tf_opt

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = self._logits, labels = self.one_hot_y)
        loss_operation = tf.reduce_mean(cross_entropy)

        #optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        optimizer = tf_opt(learning_rate = lr)
        self.training_operation = optimizer.minimize(loss_operation)

        correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print("{0:*^80}".format(" START of Training "))
        t0 = time.time()
        self._run_training(use_gpu)
        self.training_time = time.time() - t0
        print('Duration: {:.1f}s'.format(self.training_time))
        print("{0:*^80}".format(""))



    def _create_cnn(self, x, drop_out, filter_num):
        """
            two convolutional layers (16 3 × 3 filters and a stride of 1),
            each followed by ReLU activations and a max pooling layer.
            After the convolution layers we add a fully connected layer
            with 128 units and a softmax layer to do the classification.
        """

        with tf.name_scope("h_layers"):
            h_conv_pool_1 = max_pool_2x2(layer_conv(x,[3,3,1, filter_num]))
            h_conv_pool_2 = max_pool_2x2(layer_conv(h_conv_pool_1,[3, 3, filter_num, filter_num]))
            h_flatterd = tf.reshape(h_conv_pool_2, [-1, 7*7*filter_num])
            h_fc_3 = fully_connected_layer(h_flatterd, [7*7*filter_num,128])

            if drop_out:
                # Regularization  with droput
                self._drop_out_keep_prob = tf.placeholder(tf.float32)
                h_drop_out = tf.nn.dropout(h_fc_3, self._drop_out_keep_prob)
            # softmax output
            logits = fully_connected_layer(h_fc_3, [128,10], actication = None)

        return logits

    def plot_accuracy(self, ax = None, title = None, plt_validation_accuracy = True, plt_test_accuracy = True):
        '''plot the network performance'''
        if ax is None:
            ax = plt
        epoch_x = np.arange(len(self.epoch_validation_accuracy))
        handles =[]
        label = "lr: {}, filter num: {}".format(self.lr,self.filter_num)
        if plt_validation_accuracy:
            line_val, = ax.plot(epoch_x, self.epoch_validation_accuracy, label=label)
            handles.append(line_val)
        if plt_test_accuracy:
            line_train, = ax.plot(epoch_x, self.epoch_test_accuracy, label=label)
            handles.append(line_train)
        ax.set_ylabel('accuracy')
        ax.set_xlabel('epochs')
        # ax.legend(handles= handles)

        if title is not None:
            ax.set_title(title)
