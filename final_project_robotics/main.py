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
import sys
#sys.path.append("../..") # Adds higher directory to python modules path.
from framework.world import PltWorld

import numpy as np
from framework.environment import PltPolygon
from framework.test import Cleaner

#differnt maps
# from aadc.tracks.track_random_crossing import get_test_track
# from aadc.tracks.track_circle import get_test_track
# from aadc.tracks.track_chess import get_test_track
from aadc.tracks.track_empty import get_test_track

import matplotlib.pyplot as plt
from qagentcar import QAgentCar
import matplotlib.pyplot as plt


import keras
keras.backend.clear_session()

#simulation param
fig_size = (4, 4)

sim_time = 500
sim_interval_s = 0.05

print("steps: {}".format(sim_time/sim_interval_s))

#random cleaner bots param
RANDOM_CLEANER_CNT = 10
SIZE_CLEANER_CAR = 0.2

# occupancy grid map for the local agent view
grid_size_x = 50      # half to left and half to right
grid_size_y = 50 # grids points to front
grid_offset_y = grid_size_y * 0.5  # in the initial grid the car is in the center, ->grind in front of the car
grid_scale_x = 0.1  # TODO  real grid resolution is currently 1/2
grid_scale_y = grid_scale_x

# restore_wights_files = "network.h5"
restore_wights_files = None

#if test file is not None the animation is enabled and no train
test_wights_files = None
test_wights_files = "network.h5"


def helper_is_in_elements(el, elements):
    for e in elements:
        x, y, theta = el.get_position()
        if e.is_in_element([x, y], el.get_radius()):
            return True
    return False

def helper_run_game(qcar, world_size =[5,5], cnt_cleaner =1,):
    '''creats and run a world with side boarders and Cleaner agnets'''
    start_pos, word_size_real, wall_elements = get_test_track(world_size)

    world = PltWorld(animation=test_wights_files is not None    ,
                     # backgroud_color="black",
                     worldsize_max=word_size_real,
                     figsize=fig_size)

    world.add_element(qcar)
    # add cleaner at random postions
    cleaner_elements = []
    while len(cleaner_elements) < cnt_cleaner:
        cleaner = Cleaner(x=0, y=0, wheelDistance=SIZE_CLEANER_CAR, theta=0, show_path=False)
        cleaner.place_random(x_max=word_size_real[0] * 4 / 5, x_min=word_size_real[0] * 1 / 5,
                             y_max=word_size_real[1] * 4 / 5, y_min=word_size_real[1] * 1 / 5)
        cleaner.limit_movement(True,x_max=word_size_real[0] , x_min=0,
                             y_max=word_size_real[1] , y_min=0)
        #check if cleaner not colliding
        if not helper_is_in_elements(cleaner,cleaner_elements) and not helper_is_in_elements(cleaner,wall_elements):
            cleaner_elements.append(cleaner)

    # add cleaner bots and walls to world
    for el in cleaner_elements:
        world.add_element(el)
    for el in wall_elements:
        world.add_element(el)

    # start sim
    world.simulate(sim_interval_s=sim_interval_s,
                   sim_duration_s=sim_time,
                   save_animation=False,
                   ui_fps=None,
                   ui_close_window_after_sim=True)


if __name__ == "__main__":
    world_size=[3,3]
    u = QAgentCar.MAX_SPEED
    u_s = u*0.75
    u_ss = u*0.5 #side stearing
    # u_b  = -u*0.25#back
    # actions =  [[u,u,0],[u,u,0.2*np.pi],[u,u,-0.2*np.pi],[u_ss,u_ss,0.4*np.pi],[u_ss,u_ss,-0.4*np.pi],[u_b,u_b,0.4*np.pi],[u_b,u_b,-0.4*np.pi]]
    #action like: [u_right,u_left, stearing_cmd in rad]
    actions =  [[u,u,0],[u_s,u_s,0.2*np.pi],[u_s,u_s,-0.2*np.pi],[u_ss,u_ss,0.4*np.pi],[u_ss,u_ss,-0.4*np.pi]]

    qcar = QAgentCar(actions = actions,#action the agent can perform
                            x=1.5, y=1.5, theta=0.25*np.pi,radius =0.25 ,color ="darkorange",  # init car pos
                            u =[[4,5,np.pi*0.2]],# single command mode for the SimpleCarMdl in [[u_1,u_2, stearing_cmd]]
                            world_size = world_size,# world size for random postion after collisions
                            hist_len = 2,
                            restore_wights_files = restore_wights_files,
                            test_wights_files =test_wights_files,#file to load test weights, if loaded no training
                            use_history = False,# disable agent position history
                            # perseption grid sensor:
                            use_gird_history = False,
                            grid_x_size=grid_size_x, grid_y_size=grid_size_y, grid_scale_x=grid_scale_x,
                            grid_scale_y=grid_scale_y, grid_offset_y=grid_offset_y)

    helper_run_game(qcar, world_size, RANDOM_CLEANER_CNT)
