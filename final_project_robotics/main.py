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
sys.path.append("../..") # Adds higher directory to python modules path.
from simframework.framework.world import PltWorld

import numpy as np
from simframework.framework.environment import PltPolygon
from simframework.framework.test import Cleaner
from simframework.aadc.tracks.track_random_crossing import get_test_track

fig_size = (4, 4)
size_car = 0.15
sim_time = 10
sim_interval_s = 0.01

cnt_cleaner = 5

# get map made out of polygons7
start_pos, word_size, wall_elements = get_test_track()

world = PltWorld(animation=True,
                 # backgroud_color="black",
                 worldsize_max=word_size,
                 figsize=fig_size)
# add cleaner
i = 0
while i < cnt_cleaner:
    cleaner = Cleaner(x=0, y=0, wheelDistance=size_car, theta=0, show_path=True)
    cleaner.place_random(x_max=word_size[0] * 4 / 5, x_min=word_size[0] * 1 / 5,
                         y_max=word_size[1] * 4 / 5, y_min=word_size[1] * 1 / 5)
    world.add_element(cleaner)
    # todo repalce if in wall
    i += 1

# add to world
for el in wall_elements:
    world.add_element(el)

# start sim
world.simulate(sim_interval_s=sim_interval_s,
               sim_duration_s=sim_time,
               save_animation=False,
               ui_fps=None,
               ui_close_window_after_sim=True)
