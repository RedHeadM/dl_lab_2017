import numpy as np
import tensorflow as tf

# custom modules
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D

from keras import optimizers


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this script assumes you did generate your data with the get_data.py script
# you are of course allowed to change it and generate data here but if you
# want this to work out of the box first run get_data.py
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)

# 1. train
######################################
# TODO implement your training here!
# you can get the full data from the transition table like this:
#
# # both train_data and valid_data contain tupes of images and labels
# train_data = trans.get_train()
# valid_data = trans.get_valid()
#
# alternatively you can get one random mini batch line this
#
# for i in range(number_of_batches):
#     x, y = trans.sample_minibatch()
# Hint: to ease loading your model later create a model.py file
# where you define your network configuration
######################################

[train_states, train_labels] = trans.get_train()
[valid_states, valid_labels] = trans.get_valid()
print("train data shape {}",train_states.shape)
print("train data shape {}",train_labels.shape)

print("valid data shape {}",valid_states.shape)
print("valid data shape {}",valid_labels.shape)

train_shaped = train_states.reshape(train_states.shape[0], opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz, opt.hist_len)
valid_shaped = valid_states.reshape(valid_states.shape[0], opt.cub_siz*opt.pob_siz, opt.cub_siz*opt.pob_siz, opt.hist_len)

#train_shaped = tf.reshape(train_states, [-1,25, 25, 4])
train_shaped = train_shaped.astype('float32')
valid_shaped = valid_shaped.astype('float32')
num_classes = 5

input_shape = (opt.cub_siz*opt.pob_siz,opt.cub_siz*opt.pob_siz,opt.hist_len)

print(train_shaped.shape)

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
                 activation='relu',
                 input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
#              metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001),
              metrics=['accuracy'])
epochs = 10

model.fit(train_shaped, train_labels,
          batch_size=trans.minibatch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(valid_shaped, valid_labels),
          callbacks=[history])




# 2. save your trained model
model.save('my_model.h5')
