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

class DQNAgent:
    def __init__(self, state_size, action_size, use_conv):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.use_conv = use_conv
        print("use_conv",use_conv)
        if not use_conv:
            self.model = self._build_model()
            self.target_model = self._build_model()
        else:
            self.model = self._build_model_conv()
            self.target_model = self._build_model_conv()
        # needed  to _make_predict_function initialize before threading
        #https://github.com/keras-team/keras/issues/2397
        print('testing model:', self.model.predict(np.zeros((1,)+state_size)))
        print('testing target_model:', self.target_model.predict(np.zeros((1,)+state_size)))
        self.model.fit(np.zeros((1,)+state_size), np.zeros((1,5)), epochs=1, verbose=0)
        self.model.summary()
        print("state_size: ",state_size)
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    # def _build_model(self):
    #     # Neural Net for Deep-Q learning Model
    #     model = Sequential()
    #     model.add(Dense(5000, input_dim=self.state_size, activation='relu'))
    #     model.add(Dense(2000, activation='relu'))
    #     model.add(Dense(self.action_size, activation='linear'))
    #     model.compile(loss=self._huber_loss,
    #                   optimizer=Adam(lr=self.learning_rate))
    #     return model

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
        model.compile(loss='mse',
                  optimizer=Adam(lr=self.learning_rate))
        # model.compile(loss='mse',
        #                optimizer=keras.optimizers.SGD(lr=0.001),)


        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state,enable_exploration = True):
        if np.random.rand() <= self.epsilon and enable_exploration:
            return random.randrange(self.action_size)
        state = state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if self.use_conv:
                   #reshape the input frames
                  state = state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
                  #reshape the next input frames
                  next_state = next_state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
            else:
                  state = np.array([state])
                  next_state = np.array([next_state])
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.model.save_weights(name)