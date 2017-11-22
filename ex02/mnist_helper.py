#!/usr/bin/env python3
import numpy as np
#import cPickle
import _pickle as cPickle #python3.x cPickle has changed from cPickle to _pickle or use pickle
import os
import gzip

from tensorflow.examples.tutorials.mnist import input_data



def load_mnist_data(n_train_samples = 100, reshape = False):
    mnist = input_data.read_data_sets("MNIST_data_tf/", reshape=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_valid, y_valid = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels
    if n_train_samples is not None:
        train_idxs = np.random.permutation(X_train.shape[0])[:n_train_samples]
        X_train = X_train[train_idxs]
        y_train = y_train[train_idxs]
    return [X_train, y_train, X_valid, y_valid, X_test, y_test]

def __load_mnist_data(n_train_samples = 1000, reshape = False):
    Dtrain, Dval, Dtest = _mnist()
    X_train, y_train = Dtrain
    X_valid, y_valid = Dval
    X_test, y_test = Dtest
    # Downsample training data to make it a bit faster for testing this code
    train_idxs = np.random.permutation(X_train.shape[0])[:n_train_samples]
    X_train = X_train[train_idxs]
    y_train = y_train[train_idxs]
    if reshape:
        # reshape to flat vec
        X_train = X_train.reshape(X_train.shape[0], -1)
        print("Reshaped X_train size: {}".format(X_train.shape))
        X_valid = X_valid.reshape((X_valid.shape[0], -1))
        print("Reshaped X_valid size: {}".format(X_valid.shape))
        X_test = X_test.reshape((X_test.shape[0], -1))
    return [X_train, y_train, X_valid, y_valid, X_test, y_test]




def _mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')

    try:
        #test = np.array(cPickle.load(f, encoding="latin1"))
        #for x in test:
        #    print("shape: {}".format(np.shape(x)))
        #train_set =test[0]
        #valid_set =test[1]
        #test_set = test[2]
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #print ("train_set",train_set)
    #print ("valid_set",valid_set)
    #print ("test_set",test_set)
    test_x = test_set[0]
    test_y = test_set[1]
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 1, 28, 28)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 1, 28, 28)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 1, 28, 28)
    train_y = train_y.astype('int32')
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    print('... done loading data')
    return rval
