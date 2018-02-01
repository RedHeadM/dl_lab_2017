#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: csv_plt.py
    Abstract: Polt a csv file based on https://github.com/keon/deep-q-learning/blob/master/ddqn.py
    Author: Markus Merklinger
    Date created: 10/20/2017
    Date last modified: 10/20/2017
    Python Version: 3.5
'''
__version__ = "1.0.0"
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
from keras import backend as K

from  framework.utils.log import log

class DQNAgent:
    def __init__(self, state_size, action_size, use_conv):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.use_conv = use_conv
        self.last_loss_replay = 0
        self.history_len = state_size[2]
        self.memory_counter = 0
        self.memory_size = 2000
        self.batch_size = 64
        size_art = 3#reward action and terminated
        self.memory_state_size=np.prod(np.array(state_size))
        #2 * becuse two state saved
        self.memory = np.empty([self.memory_size,2*self.memory_state_size+size_art])




        log.debug("use_conv: {}".format(use_conv))
        if not use_conv:
            self.model = self._build_model()
            self.target_model = self._build_model()
        else:
            self.model = self._build_model_conv()
            self.target_model = self._build_model_conv()
        # needed  to _make_predict_function initialize before threading
        #https://github.com/keras-team/keras/issues/2397
        w = self.model.get_weights()
        log.debug('testing model: {}'.format( self.model.predict(np.zeros((1,)+state_size))))
        log.debug('testing target_model: {}'.format( self.target_model.predict(np.zeros((1,)+state_size))))
        self.model.fit(np.zeros((1,)+state_size), np.zeros((1,self.action_size)), epochs=1, verbose=0)
        self.model.set_weights(w)
        self.model.summary()
        log.debug("state_size: {}".format(state_size))
        log.debug("action_size: {}".format(action_size))
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(5000, input_dim=self.state_size, activation='relu'))
        model.add(Dense(2000, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_model_conv(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
                     activation='relu',
                     input_shape=self.state_size))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse',#loss=self._huber_loss,#
                  optimizer=Adam(lr=self.learning_rate))
        # model.compile(loss='mse',
        #                optimizer=keras.optimizers.SGD(lr=self.learning_rate),)
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, enable_exploration = True):
        if np.random.rand() <= self.epsilon and enable_exploration:
            return random.randrange(self.action_size)
        state = state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def memory_store(self, state_now, action, reward, state_next, done):
        state_now = np.reshape(state_now, [1, self.memory_state_size])
        state_next = np.reshape(state_next, [1, self.memory_state_size])
        action = np.reshape(action, [1, 1])
        reward = np.reshape(reward, [1, 1])
        done = np.reshape(done, [1, 1])

        transition = np.hstack((state_now, action, reward, state_next, done))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def replay(self):

        loss = 0.
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        ################################################################
        new_state_size = self.memory_state_size
        batch_state = batch_memory[:, :new_state_size]
        batch_action = batch_memory[:, new_state_size].astype(int)
        batch_reward = batch_memory[:, new_state_size+1]

        batch_state_next = batch_memory[:, -new_state_size-1:-1]
        batch_done = batch_memory[:, -1]

        batch_state = batch_state.reshape(self.batch_size,self.state_size[0],self.state_size[1],self.state_size[2])
        batch_state_next = batch_state_next.reshape(self.batch_size,self.state_size[0],self.state_size[1],self.state_size[2])
        q_target = self.model.predict(batch_state)
        q_next1 = self.model.predict(batch_state_next)
        q_next2 = self.target_model.predict(batch_state_next)
        batch_action_withMaxQ = np.argmax(q_next1, axis=1)
        batch_index11 = np.arange(self.batch_size, dtype=np.int32)
        q_next_Max = q_next2[batch_index11, batch_action_withMaxQ]
        # q_target[batch_index11, batch_action] = batch_reward + (1-batch_done)*self.gamma * q_next_Max
        q_target[batch_index11, batch_action] = batch_reward + self.gamma * q_next_Max
        batch_state = batch_state.reshape((self.batch_size,self.state_size[0], self.state_size[1], self.state_size[2]))
        history = self.model.fit(
            batch_state, q_target,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0)
            # log.error(history.history.keys())
            # log.error("loss".format(history.history['loss']))
        l = history.history['loss']
        loss +=np.mean(l)
        if np.isnan(l):
            log.error("loss is nan!")
        self.last_loss_replay = loss
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()
        #DEBUG prints to check loading and save
        # log.error(' zeros testing model:', self.model.predict(np.zeros((1,)+self.state_size)))
        # log.error(' ones testing model:', self.model.predict(np.ones((1,)+self.state_size)))

    def save(self, name):
        #DEBUG prints to check loading and save
        # log.error(' zeros testing model:', self.model.predict(np.zeros((1,)+self.state_size)))
        # log.error(' ones testing model:', self.model.predict(np.ones((1,)+self.state_size)))
        if np.any(np.isnan(self.model.predict(np.zeros((1,)+self.state_size)))):
            log.error("nan in predict")
        if np.any(np.isnan(self.model.predict(np.ones((1,)+self.state_size)))):
            log.error("nan in predict")
        self.model.save_weights(name,overwrite=True)
