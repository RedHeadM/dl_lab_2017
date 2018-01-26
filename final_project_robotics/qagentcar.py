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

    def __init__(self,actions,grid_x_size,grid_y_size,radius, x=0, y=0, theta=np.pi, **kwargs):
        self._actions = actions
        super().__init__(x=x, y=y, theta=theta,  # circle agent ui
                        grid_x_size=grid_x_size, grid_y_size=grid_y_size,#grid sensor
                        range=(radius + 0.02), offfset=radius + 0.01, **kwargs)  # bumper sensor
        DQNAgent.__init__(self,state_size =(grid_x_size,grid_y_size,1), action_size = len(actions))
        self.current_state = np.zeros((grid_x_size, grid_y_size,1))
        self.last_distance = 0
        self.current_u_index = 0


    def sim_step_output(self, step, dt):

        # get sensor simulated_training_data
        grid_data = self.get_last_grid_data(step)
        # sensor is configured in the way that view to front
        # next_state = grid_data.reshape((1,self._grid_size_x,self._grid_size_y,1))
        next_state = np.array([grid_data.reshape(self._grid_size_x,self._grid_size_y,1)])
        if step != 0:
            #agent.remember(state, action, reward, next_state, done)
            # print("step {} reward {}".format(step,self.get_reward()))
            self.remember(self.current_state,self.current_u_index,self.get_reward(),next_state,False)
            if step >32:
                self.replay(32)

        self.current_state = next_state
        self.current_u_index = np.argmax(self.act(self.current_state,False))
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
