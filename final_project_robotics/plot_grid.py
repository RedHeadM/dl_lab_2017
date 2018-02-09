#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: plot_grid.py
    Abstract: plots the local view of the agent: occupancy gird
    Author: Markus Merklinger
'''

import math
from sys import platform as _platform
import numpy as np

from framework.world import PltWorld
from framework.agent import PltMovingCircleAgent
from framework.sensor import BumperSensor
from aadc.simpleCarMdl import SimpleCarMdl
from aadc.perceptionGirdSensor import PerceptionGridSensor
# from aadc.tracks.track_circle import get_test_track, get_test_track_wall_polygon  # change track here
from aadc.tracks.track_random_crossing import get_test_track, get_test_track_wall_polygon  # change track here
from aadc.param.aadc_parm import AadcCarParam
from framework.test import Cleaner

from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt  # call after pltworld import


# sim params
sim_time = 10
sim_interval_s = 10
fig_size = (4, 4)

# car grid sensor param
grid_x_size = 15  # half to left and half to right
grid_y_size = 15  # grids to front beciase offest
grid_offset_y = grid_y_size * 0.5  # in the initial grid the car is in the center,
# grid_offset_y = grid_y_size * 0.4  # in the initial grid the car is in the center,
#  we setup a grid which is just the front of the car
grid_scale_x = 0.25  # 2 cm grid resolution
grid_scale_y = grid_scale_x
word_size =(5,5)


class AADCcarGridTester(PltMovingCircleAgent, SimpleCarMdl, BumperSensor, PerceptionGridSensor):
    def __init__(self, x=0, y=0, theta=np.pi, lr=1., lf=1., **kwargs):
        radius = np.amax(lr + lf)
        super().__init__(x=x, y=y, theta=theta, radius=radius,  # circle agent ui
                         lr=lr, lf=lf,   # SimpleCarMdl  pos=(x,y,theta),
                         range=(radius + 0.02), offfset=radius + 0.01, **kwargs)  # bumper sensor

    def sim_step_input(self, step, dt, environment):
        super().sim_step_input(step, dt, environment)
        col = self.collistion()
        if col != BumperSensor.NONE:
            self.change_color()

start_pos, word_size_real, wall_elements = get_test_track(word_size)

world = PltWorld(animation=True,
                 # backgroud_color="black",
                 worldsize_max=word_size_real,
                 figsize=fig_size)
car = AADCcarGridTester(x=start_pos[0], y=start_pos[1], theta=np.pi * 0.3,  # init car pos
                        u=[[1,1,0]], lr=AadcCarParam.CAR_LR, lf=AadcCarParam.CAR_LF,  # car mdl
                        grid_x_size=grid_x_size, grid_y_size=grid_y_size, grid_scale_x=grid_scale_x,
                        # perseption sensor
                        grid_scale_y=grid_scale_y, grid_offset_y=grid_offset_y)
# add to world
world.add_element(car)
for el in wall_elements:
    world.add_element(el)

#add cleaner to world and for plt later to see the grid
CLEANER_SIZE = 0.2
cleaner1 = Cleaner(x=1.5, y=3, wheelDistance=CLEANER_SIZE, theta=0, show_path=False, color="orange")
cleaner1.clear_cmds()
cleaner2 = Cleaner(x=4.5, y=4.5, wheelDistance=CLEANER_SIZE, theta=0, show_path=False,color="orange")
cleaner2.clear_cmds()
world.add_element(cleaner1)
world.add_element(cleaner2)
cleaners_plt =[]
cleaner_plt_1 = Cleaner(x=1.5, y=3, wheelDistance=CLEANER_SIZE, theta=0, show_path=False,color="darkorange")
cleaner_plt_2 = Cleaner(x=4.5, y=4.5, wheelDistance=CLEANER_SIZE, theta=0, show_path=False,color="darkorange")
cleaners_plt.append(cleaner_plt_1)
cleaners_plt.append(cleaner_plt_2)


# for plotting later the grid maps
PerceptionGridSensor.GRID_DEBUG = True

# start sim
world.simulate(sim_interval_s=sim_interval_s,
               sim_duration_s=sim_time,
               save_animation=False,
               ui_fps=None,
               ui_close_window_after_sim=True)

# get results from sim
if _platform == "linux" or _platform == "linux2":
    # linux
    plt.switch_backend('TKagg')  # TODO need under Ubuntu, problem with disabled toolbar

# get sensor simulated_training_data history
# sensor_test_data=car.get_all_grid_data()

# get sensor simulated_training_data hirsotry  where GRID_FULL detected
#get all wehre empty detected
grid_sensor_sample_pos_empty = car.get_debug_grid_sample_pos_deteted(PerceptionGridSensor.GRID_EMPTY)
x_data_empty = grid_sensor_sample_pos_empty[:, 0:1]
y_data_empty = grid_sensor_sample_pos_empty[:, 1:2]

grid_sensor_sample_pos_full = car.get_debug_grid_sample_pos_deteted(PerceptionGridSensor.GRID_FULL)
x_data_full = grid_sensor_sample_pos_full[:, 0:1]
y_data_full = grid_sensor_sample_pos_full[:, 1:2]

car_pos_data = car.get_history()
print(car._grid_sensor_data_history.shape)
# plot relusts
# fig = plt.figure(figsize=(5, 5))
# ax = plt.axes()
fig, axarr = plt.subplots(1,2, sharex=True,figsize=(10, 5))


#plt cleaners
for c in cleaners_plt:
    c.ui_init_drawables(fig,axarr[0],0,0)
    c.ui_update(fig,axarr[0],0,0)
    # c.patch.set_alpha(0.9)
    # c.line.set_alpha(0.5)


print("car_pos_data",car_pos_data[0])

#add car to plt
pos_car_gird_sample_point = Cleaner(x=1, y=1, wheelDistance=CLEANER_SIZE, theta=np.pi * 0.3, show_path=False,color="blue")
pos_car_gird_sample_point.ui_init_drawables(fig,axarr[1],0,0)
pos_car_gird_sample_point.ui_update(fig,axarr[1],0,0)
# pos_car_gird_sample_point.patch.set_alpha(0.8)
# pos_car_gird_sample_point.line.set_alpha(0.5)

# plt walls again
for g in get_test_track_wall_polygon(word_size):
    axarr[0].add_patch(plt.Polygon(g, fc='grey', alpha=0.2))
for g in get_test_track_wall_polygon(word_size):
    axarr[1].add_patch(plt.Polygon(g, fc='grey', alpha=0.2))

# plt sensor grid sample points
# with sample radius

patches_empty = []
for (x, y) in zip(x_data_empty, y_data_empty):
    circle = plt.Circle((x,y), radius=car._grid_max_cell_size*0.5)
    patches_empty.append(circle)
patches_full = []
for (x, y) in zip(x_data_full, y_data_full):
    circle = plt.Circle((x,y), radius=car._grid_max_cell_size*0.5)
    patches_full.append(circle)
# colors = 100*np.random.rand(len(patches))
pf = PatchCollection(patches_full, alpha=0.4,color ="red")
pe = PatchCollection(patches_empty, alpha=0.4,color ="green")
# p.set_array(np.array(colors))
axarr[1].add_collection(pf)
axarr[1].add_collection(pe)


# plot car movment
N = np.size(car_pos_data, axis=0)
k = np.arange(N)
theta = car_pos_data[:, 2:3]

# for plot the wanted grid oiranteation
gradx = np.cos(theta)
grady = np.sin(theta)

# plot car pos
#self._grid_max_cell_size
# axarr[1].scatter(car_pos_data[:, 0], car_pos_data[:, 1], marker='.', color='r', s=100)
# polt car oriantation with arrow
# axarr[1].quiver(car_pos_data[:, 0], car_pos_data[:, 1], gradx, grady, width=0.002, color='g',)

for a in axarr:
    a.set_ylim([-0.25,5.1])
    a.set_xlim([-0.25,5.1])

    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)

    a.xaxis.set_ticks_position('none')
    a.yaxis.set_ticks_position('none')
fig.tight_layout()
# plt.savefig(os.path.join(plt_folder, plt_file_name) + '.pdf', format='pdf', dpi=1000)#save as pdf first

plt.savefig('view.pdf', format='pdf', dpi=1000)
#plt.show()
