#!/usr/bin/env python3

import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data


# assert(len(X_train) == len(y_train))
# assert(len(X_valid) == len(y_valid))
# assert(len(X_test) == len(y_test))


from mnist_helper import load_mnist_data
from small_lenet import SmallLeNet

import os



EPOCHS = 10
BATCH_SIZE = 128


def plt_different_learning_rates(lr_array, opt, drop_out, filter_num,file_name):
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11.69,4.27))
    for lr in lr_array:
        cnn = SmallLeNet(drop_out = drop_out, filter_num =filter_num)
        cnn.train_mnist(batch_size = 128, epochs = 10, lr =lr, tf_opt = opt)
        #plt valitation
        cnn.plot_accuracy(ax[0], plt_test_accuracy = False)
        #traing error
        cnn.plot_accuracy(ax[1], plt_validation_accuracy = False)


    ax[1].legend(loc='center right')
    ax[0].legend(loc='center right')

    ax[0].set_title("validation accuracy")
    ax[1].set_title("test accuracy")
    f.savefig( file_name +'.pdf')

def plt_runtime_rates(ax, drop_out, filter_num_array, batch_size = 128):
    gpu_runtime = []
    gpu_num_param = []
    cpu_runtime = []
    cpu_num_param = []
    epochs = 1
    lr = 0.1
    for filter_num in filter_num_array:

        if filter_num <= 64:
            print("CPU")
            cnn = SmallLeNet(drop_out = drop_out, filter_num =filter_num)
            cnn.train_mnist(batch_size = batch_size, epochs = epochs, lr = lr,use_gpu =False)
            cpu_runtime.append(cnn.training_time)
            cpu_num_param.append(cnn.variables_parameter)
        print("GPU")
        cnn = SmallLeNet(drop_out = drop_out, filter_num = filter_num)
        cnn.train_mnist(batch_size = batch_size, epochs = epochs, lr = lr,use_gpu =True)
        gpu_runtime.append(cnn.training_time)
        gpu_num_param.append(cnn.variables_parameter)

    #plot
    area = 4
    ax.scatter(gpu_num_param,gpu_runtime,label="GPU",s=area, color="green", alpha=0.5)
    ax.scatter(cpu_num_param,cpu_runtime,label="CPU",s=area, color="red", alpha=0.5)
    ax.legend(loc='lower right')

    ax.set_ylabel("Runtime in s")
    ax.set_xlabel("Number of parameters")
    ax.autoscale(tight=True)





# def disa_tf_cpu(enable = True):
#     ''' cuda environ var must be set for this to work
#         note: CUDA_VISIBLE_DEVICES to the empty string to run on cpu only
#     '''
#     # run
#     if not enable:
#         os.environ["CUDA_DEVICE_ORDER"]=""
#         print("TF gpu disabled")
#     else:
#         del os.environ["CUDA_DEVICE_ORDER"]

if __name__ == "__main__":

    # cnn = SmallLeNet()
    # cnn.train_mnist(batch_size = 128, epochs = 10, lr =1e-3)
    #
    # cnn2 = SmallLeNet(drop_out = True, filter_num =16)
    # cnn2.train_mnist(batch_size = 128, epochs = 10, lr =1e-3, tf_opt = tf.train.AdamOptimizer)

    lr =[1e-1,1e-2,1e-3,1e-4]
    plt_different_learning_rates(lr, tf.train.GradientDescentOptimizer,drop_out = False, filter_num = 16,file_name="lr_no_drop_out_16_filter_gd")
    plt_different_learning_rates(lr, tf.train.GradientDescentOptimizer,drop_out = True, filter_num = 16, file_name="lr_with_drop_out_16_filter_gd")
    plt_different_learning_rates(lr, tf.train.AdamOptimizer,drop_out = True, filter_num = 16,file_name="lr_with_drop_out_16_filter_adam")
    plt.show()


    # training samples 10000 set in SmallLeNet
    filter_num =[8, 16, 32, 64, 128, 256]
    # batch_size = 64
    # f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11.69,4.27))
    # plt_runtime_rates(ax[0], drop_out = False, filter_num_array =filter_num, batch_size = batch_size)
    # ax[0].set_title("no dropout")
    # plt_runtime_rates(ax[1], drop_out = True, filter_num_array =filter_num, batch_size = batch_size)
    # ax[1].set_title("with dropout")
    # f.savefig( "runtime_train_samples_10000_batch_size_64.pdf")

    # batch_size = 16
    # f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11.69,4.27))
    # plt_runtime_rates(ax[0], drop_out = False, filter_num_array =filter_num, batch_size = batch_size)
    # ax[0].set_title("no dropout")
    # plt_runtime_rates(ax[1], drop_out = True, filter_num_array =filter_num, batch_size = batch_size)
    # ax[1].set_title("with dropout")

    f.savefig( "runtime_train_samples_10000_batch_size_16.pdf")

    #
