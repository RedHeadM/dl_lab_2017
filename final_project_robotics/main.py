#!/usr/bin/env python3
'''
    File name: qagentcar.py
    Abstract: qagent lears avoid collision with dynamic and static obstacles
    Author: Markus Merklinger
    Python Version: 3.5
'''
__version__ = "1.0.0"

import sys
import numpy as np
import argparse
from framework.world import PltWorld
from framework.environment import PltPolygon
from framework.test import Cleaner
from framework.utils.log import log
# different maps:
# from aadc.tracks.track_random_crossing import get_test_track
# from aadc.tracks.track_circle import get_test_track
# from aadc.tracks.track_chess import get_test_track
from aadc.tracks.track_empty import get_test_track
from qagentcar import QAgentCar

import matplotlib.pyplot as plt
import keras
keras.backend.clear_session()

parser = argparse.ArgumentParser(description='q-car Dynamic obstacle collision avoidance using DDQN')
parser.add_argument('--test_weights', default=None, help='.h5 weight file to run a test')
parser.add_argument('--restore_weights', default=None, help='.h5 weight file to restore')
args = parser.parse_args()

#simulation param
fig_size = (4, 4)
sim_interval_s = 0.05

#random cleaner bots param
RANDOM_CLEANER_CNT = 7
SIZE_CLEANER_CAR = 0.25
COLOR_CLEANER_CAR = "darkorange"
COLOE_QAGENT = "blue"

# occupancy grid map for the local agent view
grid_size_x = 30      # half to left and half to right
grid_size_y = 30 # grids points to front
grid_offset_y = grid_size_y * 0.3  # in the initial grid the car is in the center, ->grind in front of the car
grid_scale_x = 0.1  # TODO  real grid resolution is currently 1/2
grid_scale_y = grid_scale_x

#load weights and contiune training
restore_wights_files = args.restore_weights
#if test file is not None the animation is enabled and no train
test_wights_files   = args.test_weights


def helper_is_in_elements(el, elements):
    for e in elements:
        x, y, theta = el.get_position()
        if e.is_in_element([x, y], el.get_radius()):
            return True
    return False


def helper_run_train_game(qcar,simulation_time, world_size =[5,5], cnt_cleaner =1, animation=False):
    ''' creats and run a world with side boarders and Cleaner agnets '''
    start_pos, word_size_real, elements = get_test_track(world_size)

    world = PltWorld(name ="Training World",
                     animation = test_wights_files is not None or animation,
                     # backgroud_color="black",
                     print_stats =False,
                     worldsize_max=word_size_real,
                     figsize=fig_size)
    qcar._world_size = word_size_real
    elements.append(qcar)
    # create cleaner at random postions
    cnt_added = 0
    while cnt_added < cnt_cleaner:
        cleaner = Cleaner(x=0, y=0, wheelDistance=SIZE_CLEANER_CAR, theta=0, show_path=False,color=COLOR_CLEANER_CAR)
        cleaner.place_random(x_max=word_size_real[0] * 4 / 5, x_min=word_size_real[0] * 1 / 5,
                             y_max=word_size_real[1] * 4 / 5, y_min=word_size_real[1] * 1 / 5)
        cleaner.limit_movement(True, x_max = word_size_real[0] , x_min=0,
                                     y_max = word_size_real[1] , y_min=0)
        #check if cleaner not colliding at start pos
        if not helper_is_in_elements(cleaner, elements):
            elements.append(cleaner)
            cnt_added+=1

    # add all elements to the works
    for el in elements:
        world.add_element(el)

    # start sim
    world.simulate(sim_interval_s = sim_interval_s,
                   sim_duration_s = simulation_time,
                   save_animation = False,
                   ui_fps = None,
                   ui_close_window_after_sim = True)

def helper_validation_game(qcar,simulation_time,start_cleaner_pos, world_size =[5,5],animation=False ):
    ''' run a fixed map for vaildation/testing '''
    start_pos, word_size_real, elements = get_test_track(world_size)

    world = PltWorld(name = "Validation World",
                    print_stats =False,
                    animation = animation,
                    worldsize_max=word_size_real,
                    figsize=fig_size)

    # Enable train mode
    # qcar.enabled_test_mode(False)
    elements.append(qcar)
    # create cleaner at FIXED postions
    for pos in start_cleaner_pos:
        cleaner = Cleaner(x=pos[0], y=pos[1], wheelDistance=0.2, theta=pos[2], show_path=False,color=COLOR_CLEANER_CAR)
        cleaner.limit_movement(True, x_max = word_size_real[0] , x_min=0,
                                     y_max = word_size_real[1] , y_min=0)
        #check if cleaner not colliding at start pos
        elements.append(cleaner)
        if not helper_is_in_elements(cleaner, elements):
            log.warning("WARNING cleaner start pos: collision!")

    # add all elements to the works
    for el in elements:
        world.add_element(el)

    # start sim
    world.simulate(sim_interval_s = sim_interval_s,
                   sim_duration_s = simulation_time,
                   save_animation = False,
                   ui_fps = None,
                   ui_close_window_after_sim = True)
    if len(qcar.test_run_collision_steps):
        # first_run =np.max(qcar.test_run_collision_steps)#best run
        first_run =qcar.test_run_collision_steps[0]
    else:
        print("valid run with NO COLLISION; qcar._init_pos: ",qcar._init_pos)
        first_run = 100 # no collision in the run
    return first_run
    # return np.mean(qcar.test_run_collision_steps)

def helper_points_on_circle(r, angel=2. * np.pi, n=10, origin_x=0., origin_y=0., angel_start=0):
    return [(np.cos(angel/n*x)*r+origin_x,np.sin(angel/n*x)*r+origin_y) for x in range(n+1)]

def run_validation(qcar,world_size,animation=False):
    '''run a vaildation for a fixed map with differnt start postions'''
    agent_init_pos = [[1.5,1.5,0.3*np.pi], [3.,4.5,0],[3.5,4.5,1.2*np.pi], [4.5,1.5,0.3*np.pi], [1.5,1.5,0.4*np.pi]]

    # agent_init_pos = []
    cleaner_start_pos = [[1,1,np.pi],[3.5,3.5,1.25*np.pi],[5,5,1.25*np.pi],[1,5,1.75*np.pi],[3,1,0.7*np.pi]]
    training_steps = qcar._agent_vaild_training_steps
    steps_sum = 0.
    sim_time = 15.
    word_size_real = np.array(world_size) * 2
    r = word_size_real[0]/4
    n = 5
    cpoints = helper_points_on_circle(r,n=n,origin_x=word_size_real[0]/2,origin_y=word_size_real[1]/2)
    for i, [x, y] in enumerate(cpoints):
        agent_init_pos.append([x,y,i*((-1+2*i%2)*2*np.pi/n)])

    runs = 0.
    for pos in agent_init_pos:
        qcar._init_pos  = pos
        s = helper_validation_game(qcar, sim_time,cleaner_start_pos, world_size,animation=animation)
        # if s >10 and s !=999:
            #vaild stating postions if more the 10 steps
            # runs+=1
        steps_sum+=s
    return [training_steps,steps_sum/len(agent_init_pos)]
    # return [training_steps,steps_sum/runs]


if __name__ == "__main__":
    world_size=[4,4]
    word_size_real = np.array(world_size) * 2
    u = QAgentCar.MAX_SPEED
    u_s = u*0.5
    u_ss = u*0.2

    actions = [[u,u,0],[u_s,u_s,0.2*np.pi],[u_s,u_s,-0.2*np.pi],[u_ss,u_ss,0.6*np.pi],[u_ss,u_ss,-0.6*np.pi]]
    qcar = QAgentCar(actions = actions,#action the agent can perform
                            x=1.5, y=1.5, theta=0.25*np.pi,radius =0.25 ,color =COLOE_QAGENT,  # init car pos
                            u =[[4,5,np.pi*0.2]],# single command mode for the SimpleCarMdl in [[u_1,u_2, stearing_cmd]]
                            world_size = word_size_real,# world size for random postion after collisions
                            hist_len = 2,
                            restore_wights_files = restore_wights_files,
                            test_wights_files =test_wights_files,#file to load test weights, if loaded no training
                            use_history = False,# disable agent position history
                            # perseption grid sensor:
                            use_gird_history = False,
                            grid_x_size=grid_size_x, grid_y_size=grid_size_y, grid_scale_x=grid_scale_x,
                            grid_scale_y=grid_scale_y, grid_offset_y=grid_offset_y)

    # singel training run
    # if test_wights_files:
    #     # qcar.enabled_test_mode(True)
    #     avg_steps = run_validation(qcar,world_size,animation= True)
    #     print("avg_steps: ",avg_steps)
    #     # helper_run_train_game(qcar,50, world_size, RANDOM_CLEANER_CNT)
    #
    # else:
    #     helper_run_train_game(qcar,600, world_size, RANDOM_CLEANER_CNT)

    # VALIDATION:
    if test_wights_files:
        qcar.enabled_test_mode(True)
        avg_steps = run_validation(qcar,world_size,animation= False)
        print("avg_steps: ",avg_steps)
    else:
        #TRAIN the agent
        #fill agent memory
        helper_run_train_game(qcar,80, world_size, RANDOM_CLEANER_CNT)
        results_train_step = []
        results_no_collisions_steps = []

        # qcar.enabled_test_mode(False)
        for i in range(100):
            #Enable train mode
            qcar.enabled_test_mode(False)
            helper_run_train_game(qcar,1, world_size, RANDOM_CLEANER_CNT)
            qcar.enabled_test_mode(True)
            if qcar._agent_vaild_training_steps:
                avg_steps = run_validation(qcar,world_size)
                print("i: {}. training step {}: vlidation avg steps unitll collision: {}".format(i,avg_steps[0],avg_steps[1]))
                results_no_collisions_steps.append(avg_steps[1])
                results_train_step.append(avg_steps[0])

        plt.close('all')
        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes()

        ax.plot(results_train_step,results_no_collisions_steps)
        plt.savefig("val" + '.pdf', format='pdf', dpi=1000)#save as pdf first
        plt.show()
