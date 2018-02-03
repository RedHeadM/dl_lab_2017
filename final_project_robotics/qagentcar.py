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
import collections

from  framework.utils.log import log


# define class
class QAgentCar(PltMovingCircleAgent, SimpleCarMdl, BumperSensor, PerceptionGridSensor,DQNAgent):
    ''' agent run in the simframework: action take place in the simulation output stage'''
    DEBUG = True
    MAX_SPEED = 1.

    def __init__(self,actions,grid_x_size,grid_y_size,radius, world_size, x=0, y=0, theta=np.pi, use_conv=True,hist_len = 2,test_wights_files = None, restore_wights_files = None,save_file="network.h5", **kwargs):
        self._actions = actions
        super().__init__(x=x, y=y, theta=theta,radius=radius,  # circle agent ui
                        grid_x_size=grid_x_size, grid_y_size=grid_y_size,#grid sensor
                        range=(radius + 0.05), offfset=radius + 0.01, **kwargs)  # bumper sensor
        self._init_pos =[x,y,theta]
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
        self._world_size = world_size
        self._steps_since_last_collision = 0 # cnt the steps for print outs and vaild trainning samples
        self._agent_memory_added = 0
        self._agent_vaild_training_steps = 0 # training steps: enough data was in the state with history to add a training sample
        self._save_file = save_file
        if restore_wights_files is not None:
            log.info("weight restored: " + restore_wights_files)
            self.qagent.load(restore_wights_files)
        if test_wights_files is not None:
            self.qagent.load(test_wights_files)
            self.test_enabled = True
            log.info("QAgentCar in TEST MODE! loading " + test_wights_files)
        else:
            self.test_enabled = False

    def enabled_test_mode(enable):
        self.test_enabled = enable

    def sim_init(self, simulation_duration, dt):
        super().sim_init(simulation_duration, dt)
        self._steps_since_last_collision =0

    def sim_step_output(self, step, dt):
        self._steps_since_last_collision += 1

        # get data from the PerceptionGridSensor
        # a occupancy grid map form a local agent view
        # the grid and history is the current next state of the agent
        grid_data = self.get_last_grid_data(step)
        #DEBUG prints
        # if (np.any(grid_data)):
        #     log.info("collsion in grid_data")
        # else:
        #     log.info("NO collsion in grid_data")
        next_state =grid_data.reshape(-1)
        self._append_to_hist(self._next_state_with_history, next_state)

        #wait untill agent has valid data is history after jump
        if self._steps_since_last_collision >= self.qagent.history_len and not self.test_enabled:
            # if self._steps_since_last_collision == self.qagent.history_len:
                # self.change_color("orange")

            self._agent_memory_added += 1
            #ADD the data to the memory of the agent and replay
            next_reward = self.get_reward()
            is_terminate_state = False # here always false since next_state can be used
            #is_terminate_state = self.collistion() != BumperSensor.NONE
            self.qagent.memory_store(self._state_with_history.reshape(-1),self.current_u_index,next_reward,self._next_state_with_history.reshape(-1),is_terminate_state)
            if self._agent_memory_added > self.qagent.memory_size:
                self._agent_vaild_training_steps += 1
                self.qagent.replay()#train the agent

        #Let the agent act on the current state
        self._state_with_history = np.copy(self._next_state_with_history)
        #no expporation in action if self.test_enabled
        if not self.test_enabled and self._agent_memory_added > self.qagent.memory_size:
            self.current_u_index= self.qagent.act_random()
        else:
            self.current_u_index = self.qagent.act(self._state_with_history,enable_exploration = not self.test_enabled)

        next_cmd = self._actions[self.current_u_index]


        #update_target_model and save
        if step % 50 == 0 and step !=0 and not self.test_enabled:
                self.qagent.update_target_model()
                self.qagent.save(self._save_file)
                log.info("saved to : {}".format(self._save_file))

        # check for collision
        if self.collistion() != BumperSensor.NONE:
            # self.change_color("orange")
            if self._steps_since_last_collision >= self.qagent.history_len:
                log.info("collistion! setps since last collision: {} train_stesp {}, loss sum last batch: {} epsilon: {:.2}".format(self._steps_since_last_collision,self._agent_vaild_training_steps,self.qagent.last_loss_replay,self.qagent.epsilon))
            self._steps_since_last_collision = 0 #reset counter and fill history after jump
            super().sim_step_output(step, dt)
            #reset the agent at start pos
            # self.set_postion(self._init_pos)
            self.place_ramdom_in_world()
        else:
            # set the agent steering cmd:
            self.sim_output_set_next_u(next_cmd)
            super().sim_step_output(step, dt)

    def place_ramdom_in_world(self):
        self.place_random(x_max=self._world_size[0] * 4 / 5, x_min=self._world_size[0] * 1 / 5,
                     y_max=self._world_size[1] * 4 / 5, y_min=self._world_size[1] * 1 / 5)

    def get_reward(self):
        current_moved_distance = self.get_moved_dist()
        dist_reward = current_moved_distance - self.last_distance
        if dist_reward ==0:
            log.warning("dist_reward is zero!")
        reward_stright = 0.
        if self.u[0][2] == 0:
            reward_stright = 1. #last stearing angle was 0

        self.last_distance = current_moved_distance
        if self.collistion() != BumperSensor.NONE:
            return -10.
        else:
            # log.info("dist_reward {}".format(dist_reward))
            return dist_reward * 10. + reward_stright

    def _append_to_hist(self,state, obs):
        """
        Add observation to the state with history.
        """
        for i in range(state.shape[0]-1):
            state[i, :] = state[i+1, :]#TODO np.roll
        state[-1, :] = obs
