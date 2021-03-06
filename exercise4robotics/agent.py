import random
from time import time
import numpy as np
import keras
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Conv2D,Flatten,Dense
#http://fizzylogic.nl/2017/05/08/monitor-progress-of-your-keras-based-neural-network-using-tensorboard/
#from keras.callbacks import TensorBoard
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
#start with $ tensorboard --logdir=logs/

class QMazeAgent:
    def __init__(self, state_size, action_size,use_conv):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 10.0#1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 1e-5#0.00005#1e-6#0.0005
        self._use_conv = use_conv
        if not use_conv:
            self.model = self._build_model()
            self.target_model = self._build_model()
        else:
            self.model = self._build_model_conv()
            self.target_model = self._build_model_conv()
        print("state_size: ",state_size)
        self.update_target_model()
        history = self.model.fit(np.zeros((1,)+state_size), np.zeros((1,5)), epochs=1, verbose=0)

        # print(history.history.keys())
        print("loss",history.history['loss'])

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        #model.compile(loss=self._huber_loss,
        #              optimizer=Adam(lr=self.learning_rate))
        # model.compile(loss='mse',
        #                optimizer=Adam(lr=self.learning_rate))
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
        model.compile(loss='mse',
                  optimizer=Adam(lr=self.learning_rate))
        # model.compile(loss='mse',
        #                optimizer=keras.optimizers.SGD(lr=0.001),)


        return model

    def _append_to_hist(state, obs):
        """
        Add observation to the state.
        """
        for i in range(state.shape[0]-1):
            state[i, :] = state[i+1, :]
        state[-1, :] = obs

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, enable_exploration = True):
        if np.random.rand() <= self.epsilon and enable_exploration:
            return random.randrange(self.action_size)
        if self._use_conv:
               #reshape the input frames
              state = state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
        print("state.shape",state.shape)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def train(self, minibatch, change_epsilon = True):
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = minibatch

        for state, action, next_state, reward, done in zip(state_batch, action_batch, next_state_batch, reward_batch, terminal_batch):
            #state = np.array([state.reshape(-1)])
            #next_state = np.array([next_state.reshape(-1)])
            if self._use_conv:
                   #reshape the input frames
                  state = state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
                  #reshape the next input frames
                  next_state = next_state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
            else:
                  state = np.array([state])
                  next_state = np.array([next_state])
            # np.array([state_with_history.reshape(-1)])
            action = np.argmax(action)
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            history = self.model.fit(state, target, epochs=1, verbose=0)#, callbacks=[tensorboard])
            # print(history.history.keys())
            # print("loss",history.history['loss'])
        if self.epsilon > self.epsilon_min and change_epsilon:
            self.epsilon *= self.epsilon_decay

    def load(self, file_name):
        self.model.load_weights(file_name)
        print("agent loaded weights"+ file_name)
        self.update_target_model()

    def save(self, file_name):
        self.model.save_weights(file_name)
        print("agent saved weights"+ file_name)
