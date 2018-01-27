#!/usr/bin/env python3
'''
    File name: csv_plt.py
    Abstract: Polt a csv file
    Author: Markus Merklinger
    Date created: 10/20/2017
    Date last modified: 10/20/2017
    Python Version: 3.5
'''
__version__ = "1.0.0"
import math
from sys import platform as _platform
import numpy as np

from framework.world import PltWorld
from framework.agent import PltMovingCircleAgent
from framework.environment import PltPolygon
from framework.sensor import BumperSensor

from aadc.simpleCarMdl import SimpleCarMdl
from aadc.perceptionGirdSensor import PerceptionGridSensor
from aadc.tracks.track_circle import get_test_track, get_test_track_wall_polygon  # change world here
from aadc.param.aadc_parm import AadcCarParam

import matplotlib.pyplot as plt  # call after pltworld import

from qagent import DQNAgent





# define class
class QAgentCar(PltMovingCircleAgent, SimpleCarMdl, BumperSensor, PerceptionGridSensor,DQNAgent):
    DEBUG = True
    CONST_SPEED = 1

    def __init__(self,actions,grid_x_size,grid_y_size,radius, x=0, y=0, theta=np.pi, use_conv=True,hist_len = 2, **kwargs):
        self._actions = actions
        super().__init__(x=x, y=y, theta=theta,  # circle agent ui
                        grid_x_size=grid_x_size, grid_y_size=grid_y_size,#grid sensor
                        range=(radius + 0.02), offfset=radius + 0.01, **kwargs)  # bumper sensor

        if use_conv:
            assert hist_len >= 2
            self._state_size = (grid_x_size, grid_y_size, hist_len)
            self.qagent = DQNAgent(state_size =self._state_size, action_size = len(actions),use_conv=use_conv)
        else:
            self._state_size = (grid_x_size* grid_y_size *hist_len)
            self.qagent = DQNAgent(state_size =self._state_size, action_size = len(actions),use_conv=use_conv)

        self.last_distance = 0
        self.current_u_index = 0
        self._state_with_history = np.zeros((hist_len,(grid_x_size* grid_y_size)))
        self._next_state_with_history = np.copy(self._state_with_history)

    def sim_step_output(self, step, dt):

        # get sensor simulated_training_data
        grid_data = self.get_last_grid_data(step)
        # sensor is configured in the way that view to front
        # next_state = grid_data.reshape((1,self._grid_size_x,self._grid_size_y,1))


        next_state =grid_data.reshape(-1)
        self._append_to_hist(self._next_state_with_history, next_state)
        print("self._next_state_with_history.shape:  ",self._next_state_with_history.shape)
        if step != 0:
            #agent.remember(state, action, reward, next_state, done)
            # print("step {} reward {}".format(step,self.get_reward()))
            self.qagent.remember(self._state_with_history.reshape(-1),self.current_u_index,self.get_reward(),self._next_state_with_history.reshape(-1),False)
            if step >32:
                self.qagent.replay(32)
        else:
            print("************************")
            print("state input shape: ",next_state.shape)

        self._state_with_history = np.copy(self._next_state_with_history)
        self.current_u_index = self.qagent.act(self._state_with_history,False)
        next_u = [self.CONST_SPEED, self.CONST_SPEED, self._actions[self.current_u_index]]

        # change the color if lane is touched
        if self.collistion() != BumperSensor.NONE:
            self.change_color()
        # set steering:
        self.sim_output_set_next_u(next_u)
        super().sim_step_output(step, dt)

    def get_reward(self):
        current_moved_distance = self.get_moved_dist()
        dist_reward = current_moved_distance - self.last_distance
        self.last_distance = current_moved_distance
        if self.collistion() != BumperSensor.NONE:
            return -100
        else:
            return dist_reward

    def _append_to_hist(self,state, obs):
        """
        Add observation to the state.
        """
        for i in range(state.shape[0]-1):
            state[i, :] = state[i+1, :]
        state[-1, :] = obs
