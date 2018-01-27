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
keras.backend.clear_session()#TODO

fig_size = (4, 4)
size_car_cleaner = 0.2
sim_time = 100
sim_interval_s = 0.05
print("steps: {}".format(sim_time/sim_interval_s))
cnt_cleaner = 1
# car grid sensor param
grid_size_x = 25  # half to left and half to right
grid_size_y = 25  # grids to front beciase offest
grid_offset_y = grid_size_y * 0.5  # in the initial grid the car is in the center, ->grind in front of the car
grid_scale_x = 0.1  # 5 cm grid resolution
grid_scale_y = grid_scale_x



def run_game(qcar):
    # get map made out of polygons7
    start_pos, word_size, wall_elements = get_test_track(world_size=[5,5])

    world = PltWorld(animation=True,
                     # backgroud_color="black",
                     worldsize_max=word_size,
                     figsize=fig_size)


    world.add_element(qcar)
    # add cleaner
    i = 0

    while i < cnt_cleaner:
        cleaner = Cleaner(x=0, y=0, wheelDistance=size_car_cleaner, theta=0, show_path=False)
        cleaner.place_random(x_max=word_size[0] * 4 / 5, x_min=word_size[0] * 1 / 5,
                             y_max=word_size[1] * 4 / 5, y_min=word_size[1] * 1 / 5)
        cleaner.limit_movement(True,x_max=word_size[0] , x_min=0,
                             y_max=word_size[1] , y_min=0)
        world.add_element(cleaner)
        # todo repalce if in wall
        i += 1
        test_cleaner =cleaner
    # add to world
    for el in wall_elements:
        world.add_element(el)

    # start sim
    world.simulate(sim_interval_s=sim_interval_s,
                   sim_duration_s=sim_time,
                   save_animation=False,
                   ui_fps=None,
                   ui_close_window_after_sim=True)



def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

if __name__ == "__main__":

    qcar = QAgentCar(actions =[0,0.2*np.pi,-0.2*np.pi,0.4*np.pi,-0.4*np.pi],
                            x=1.5, y=1.5, theta=0.25*np.pi,radius =0.3,  # init car pos
                            u =[[4,5,np.pi*0.2]],# single command mode
                            use_history =False,
                            test_wights_files ="network.h5",
                            grid_x_size=grid_size_x, grid_y_size=grid_size_y, grid_scale_x=grid_scale_x,
                            # perseption sensor
                            grid_scale_y=grid_scale_y, grid_offset_y=grid_offset_y)
    run_game(qcar)

    # print("size qcar: {}".format(get_size(qcar)))
    # print("size cleaner: {}".format(get_size(qcar)))
    # print("hist len size qacar: {}".format(len(qcar.history)))
    # print("hist len size cleaner: {}".format(len(cleaner.history)))
